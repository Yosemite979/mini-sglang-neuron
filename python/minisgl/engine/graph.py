from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch
import torch_xla.core.xla_model as xm
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .config import EngineConfig

logger = init_logger(__name__)


def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    mem_info_dict = xm.get_memory_info()
    return mem_info_dict["bytes_limit"] - mem_info_dict["bytes_used"]


class GraphRunner:
    def __init__(
        self,
        config: EngineConfig,
        page_table: torch.Tensor,
        max_seq_len: int,
        num_pages: int,
        device: torch.device,
        dummy_page: int,
        dummy_req: Req,
    ) -> None:
        from minisgl.neuron import NeuronInputBuilder, NeuronLoadConfig, get_neuron_model

        self.device = device
        self.dummy_page = dummy_page
        self.dummy_req = dummy_req
        self.page_table = page_table
        self.max_seq_len = max_seq_len
        self.compile_only = False

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
            max_model_len=max_seq_len,
            max_extend_tokens=config.max_extend_tokens,
            block_size=config.page_size,
            num_blocks=num_pages,
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
                self.compile_only = True
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

        self.neuron_input_builder = NeuronInputBuilder(self.page_table, self.max_seq_len)


    def pad_batch(self, batch: Batch) -> int:
        max_batch_size = self.model.neuron_config.batch_size
        padded_size = max_batch_size if max_batch_size > batch.size else batch.size
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)
        return batch.padded_size - batch.size

    def forward(self, batch: Batch) -> torch.Tensor:
        model_input = self.neuron_input_builder.build(batch)
        return self.model.forward(
            input_ids=model_input.input_tokens,
            position_ids=model_input.position_ids,
            input_block_ids=model_input.input_block_ids,
            slot_mapping=model_input.slot_mapping,
            block_tables=model_input.block_tables,
            full_context_lens=model_input.full_context_lens,
            computed_context_lens=model_input.computed_context_lens,
        )
