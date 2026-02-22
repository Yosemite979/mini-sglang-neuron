from __future__ import annotations

from datetime import timedelta
from typing import Dict, NamedTuple, Tuple
import gc
import os
import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met


from minisgl.attention import create_attention_backend
from minisgl.attention.neuron import NeuronAttnBackend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_hf_weight
from minisgl.utils import divide_even, init_logger, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: object


def create_page_table(shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.int32, device=device)


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32


class Engine:
    def __init__(self, config: EngineConfig):
        self.model_config = config.model_config
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)

        self.stream = None # XLA does not support stream
        self.dtype = config.dtype
        self.use_neuron_model = config.use_neuron_model

        self.device = torch.device("cpu") # Use cpu for management purpose
        
        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # load model and determine number of pages
        """
        set_rope_device(self.device)
        """
        self.num_pages = self.dummy_page = self._determine_num_pages(init_free_memory, config)
        #if self.use_neuron_model:
        #    self.kv_cache = self.model.get_kv_caches()
        #else:
        #    self.kv_cache = create_kvcache(
        #        model_config=config.model_config,
        #        num_pages=self.num_pages + 1,  # +1 for dummy page
        #        device=self.device,
        #        dtype=self.dtype,
        #    )
        # NOTE: make page table 128 aligned (32 * sizeof(int32) == 128 bytes)
        self.max_seq_len = _align_up_32(min(config.max_seq_len, self.num_pages))
        # The last page (with index `self.num_pages`) is reserved for dummy requests, which should never be allocated to real requests. This simplifies the handling of padded dummy requests and chunked requests that require padding.
        self.page_table = create_page_table(  # + 1 for dummy request
            (config.max_running_req + 1, self.max_seq_len),
            device=self.device,
        )
        #if self.use_neuron_model:
        #    self.attn_backend = NeuronAttnBackend(self.page_table)
        #else:
        #    self.attn_backend = create_attention_backend(
        #        config.attention_backend,
        #        config.model_config,
        #        self.kv_cache,
        #        self.page_table,
        #    )

        #if self.use_neuron_model:
        from minisgl.neuron import NeuronInputBuilder, NeuronLoadConfig, get_neuron_model

        compile_kwargs = {}
        if config.hlo_debug:
            compile_kwargs["debug"] = True
        if config.compile_dry_run:
            compile_kwargs["dry_run"] = True

        
        load_cfg = NeuronLoadConfig(
            model_path=config.model_path,
            hf_config=config.hf_config,
            tp_degree=config.tp_info.size,
            max_batch_size=config.max_running_req,
            max_model_len=self.max_seq_len,
            max_extend_tokens=config.max_extend_tokens,
            block_size=config.page_size,
            num_blocks=self.num_pages,
            override_neuron_config=config.neuron_config_overrides,
            compile_kwargs=compile_kwargs or None,
        )
        self.model = get_neuron_model(load_cfg, init_only=config.compiled_model_path is not None)
        if config.compiled_model_path is not None:
            compiling_start_time = time.monotonic()
            if not config.skip_compile and not config.on_cpu:
                logger.info_rank0("Compiling and saving model...")
                self.model.compile(
                    config.compiled_model_path,
                    debug=config.hlo_debug,
                    dry_run=config.compile_dry_run,
                )
                total_compiling_time = time.monotonic() - compiling_start_time
                logger.info_rank0(f"Compiling and tracing time: {total_compiling_time} seconds")
            else:
                logger.info_rank0("Skipping model compilation")

            if config.enable_torch_dist:
                torch.distributed.barrier()

            if config.compile_only or config.compile_dry_run:
                logger.info_rank0("Compile-only mode enabled; skipping model load and engine init.")
                return

            loading_start_time = time.monotonic()
            if not config.on_cpu:
                logger.info_rank0("Loading model to Neuron...")
                self.model.load(config.compiled_model_path)
            else:
                logger.info_rank0("Loading model to CPU...")
                if hasattr(self.model, "to_cpu"):
                    self.model.to_cpu()
                else:
                    logger.warning("Model does not implement to_cpu(); keeping current device placement.")
            model_loading_time = time.monotonic() - loading_start_time
            logger.info_rank0(f"Total model loading time: {model_loading_time} seconds")

        self.attn_backend = None
        self.ctx = Context(page_size=1, attn_backend=self.attn_backend)
        set_global_ctx(self.ctx)
        self.sampler = Sampler(self.device, self.model_config.vocab_size)
        self.neuron_input_builder = (
            NeuronInputBuilder(self.page_table, self.max_seq_len) if self.use_neuron_model else None
        )

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # cuda graph related
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        self.page_table[self.dummy_req.table_idx].fill_(self.dummy_page)

        #if self.use_neuron_model:
        #    self.graph_runner = _NoGraphRunner(self.dummy_req)
        #else:
        #    self.graph_runner = GraphRunner(
        #        stream=self.stream,
        #        device=self.device,
        #        model=self.model,
        #        attn_backend=self.attn_backend,
        #        cuda_graph_bs=config.cuda_graph_bs,
        #        cuda_graph_max_bs=config.cuda_graph_max_bs,
        #        free_memory=init_free_memory,
        #        max_seq_len=self.max_seq_len,
        #        vocab_size=self.model_config.vocab_size,
        #        dummy_req=self.dummy_req,
        #    )
    def pad_batch(self, batch: Batch) -> int:
        max_batch_size = self.model.neuron_config.batch_size
        padded_size = (  # choose the first available batch size
            max_batch_size 
            if max_batch_size > batch.size 
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)
        logger.error(f"xinux - pad_batch: batch.size={batch.size}, extra_padded_size={padded_size-batch.size}")
        return batch.padded_size - batch.size
    
    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        torch.distributed.init_process_group(
            backend="gloo",
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=config.distributed_timeout),
            init_method=config.distributed_addr,
        )
        tp_cpu_group = torch.distributed.group.WORLD
        assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            return {
                k: v.to(self.dtype)
                for k, v in load_hf_weight(config.model_path, self.device).items()
            }

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * self.model_config.head_dim
            * divide_even(self.model_config.num_kv_heads, config.tp_info.size)
            * config.page_size
            * self.dtype.itemsize
            * self.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            # In NxDI, the page number should be pre-allocated before the model weight is loaded.
            # Thus, we are not able to pre-determine the max available memory after the model is loaded.
            # For now, we just assume a hard-coded value as a workaround.
            num_pages = config.max_seq_len * config.max_running_req

            #model_memory = old_free_memory - new_free_memory
            #available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            #num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-tokens"
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_pages} pages for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        torch_xla.sync()
        xm.wait_device_ops()
        gc.collect()
        met.clear_metrics()

        device_list = xm.get_xla_supported_devices()
        free_memory_list = [get_free_memory(device) for device in device_list]
        max_free_memory, min_free_memory = max(free_memory_list), min(free_memory_list)

        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        with self.ctx.forward_batch(batch):
            assert self.neuron_input_builder is not None
            model_input = self.neuron_input_builder.build(batch)
            logits = self.model.forward(
                input_ids=model_input.input_tokens,
                position_ids=model_input.position_ids,
                input_block_ids=model_input.input_block_ids,
                slot_mapping=model_input.slot_mapping,
                block_tables=model_input.block_tables,
                full_context_lens=model_input.full_context_lens,
                computed_context_lens=model_input.computed_context_lens,
            )

        for req in batch.reqs:
            req.complete_one()

        if self.use_neuron_model:
            next_tokens_cpu = _sample_cpu(logits[: batch.size].to("cpu"), batch.reqs)
            next_tokens_gpu = next_tokens_cpu.to(self.device)
        else:
            next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
            next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        if self.use_neuron_model:
            xm.mark_step()
            copy_done_event = _NoOpEvent()
        else:
            copy_done_event = torch.cuda.Event()
            copy_done_event.record(self.stream)
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        #self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


class _NoOpEvent:
    def synchronize(self) -> None:
        return


class _NoGraphRunner:
    def __init__(self, dummy_req: Req) -> None:
        self.dummy_req = dummy_req

    def pad_batch(self, batch: Batch) -> int:
        batch.padded_reqs = batch.reqs
        return 0


def _sample_cpu(logits: torch.Tensor, reqs: list[Req]) -> torch.Tensor:
    output = torch.empty((len(reqs),), dtype=torch.int32)
    for i, req in enumerate(reqs):
        params = req.sampling_params
        row = logits[i].float()
        if not params.is_greedy and params.temperature > 0:
            row = row / params.temperature
        if params.top_k and params.top_k > 0:
            topk = torch.topk(row, k=min(params.top_k, row.numel()))
            probs = torch.softmax(topk.values, dim=-1)
            idx = torch.multinomial(probs, 1)
            token = topk.indices[idx]
        elif params.top_p < 1.0:
            probs = torch.softmax(row, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative <= params.top_p
            mask[0] = True
            filtered_probs = sorted_probs[mask]
            filtered_idx = sorted_idx[mask]
            filtered_probs = filtered_probs / filtered_probs.sum()
            idx = torch.multinomial(filtered_probs, 1)
            token = filtered_idx[idx]
        else:
            token = torch.argmax(row, dim=-1)
        output[i] = token.item()
    return output
