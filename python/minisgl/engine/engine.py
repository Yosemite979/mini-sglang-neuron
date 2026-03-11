from __future__ import annotations

from datetime import timedelta
import time
from typing import NamedTuple, Tuple
import gc

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met


from minisgl.core import Batch, Req
from minisgl.distributed import set_tp_info
from minisgl.utils import divide_even, init_logger

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    next_tokens_cpu: torch.Tensor


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

        self.device = torch.device("cpu") # Use cpu for management purpose
        
        self.tp_cpu_group = self._init_communication(config)
        init_free_memory = self._sync_get_memory()[1]
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        self.num_pages = self._determine_num_pages(init_free_memory, config)
        self.dummy_page = 0
        # NOTE: make page table 128 aligned (32 * sizeof(int32) == 128 bytes)
        self.max_seq_len = _align_up_32(min(config.max_seq_len, self.num_pages))
        # Page ID 0 is reserved for dummy/padded requests. Real cache pages use IDs 1..num_pages.
        self.page_table = create_page_table(  # + 1 for dummy request
            (config.max_running_req + 1, self.max_seq_len),
            device=self.device,
        )

        self.sampler = Sampler(self.device, self.model_config.vocab_size)

        # Dummy request/page for padded scheduling.
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
        self.graph_runner = GraphRunner(
            config=config,
            page_table=self.page_table,
            max_seq_len=self.max_seq_len,
            num_pages=self.num_pages + 1,
            device=self.device,
            dummy_page=self.dummy_page,
            dummy_req=self.dummy_req,
        )
        self.model = self.graph_runner.model
        if self.graph_runner.compile_only:
            return

        post_free_memory = self._sync_get_memory()[0]
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

    def pad_batch(self, batch: Batch) -> int:
        return self.graph_runner.pad_batch(batch)
    
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
            
            # TODO: find a better way to determine the number of pages. The current method is too conservative and may lead to under-utilization of memory.
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
        forward_start = time.perf_counter()
        logits = self.graph_runner.forward(batch)
        model_elapsed = (time.perf_counter() - forward_start) * 1000
        logger.debug(
            "[PERF] model_execution: %.2fms [phase=%s batch=%d]",
            model_elapsed,
            batch.phase,
            batch.size,
        )

        for req in batch.reqs:
            req.complete_one()

        # The logits from NxDI are already on CPU, so sampler can consume them directly.
        sample_start = time.perf_counter()
        next_tokens_cpu = self.sampler.sample(logits[: batch.size], args)
        sample_elapsed = (time.perf_counter() - sample_start) * 1000
        logger.debug(
            "[PERF] sample_tokens: %.2fms [phase=%s batch=%d]",
            sample_elapsed,
            batch.phase,
            batch.size,
        )

        # There is no async copy in this case, but we keep the event for interface consistency and future extension.
        xm.mark_step()  # Ensure all XLA operations are finished before sampling
        total_elapsed = (time.perf_counter() - forward_start) * 1000
        logger.debug(
            "[PERF] forward_batch total: %.2fms [phase=%s batch=%d]",
            total_elapsed,
            batch.phase,
            batch.size,
        )

        return ForwardOutput(next_tokens_cpu)

    def shutdown(self) -> None:
        torch.distributed.destroy_process_group()
