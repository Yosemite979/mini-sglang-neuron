from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    max_position: int


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 256
    page_size: int = 1
    memory_ratio: float = 0.9
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    neuron_config_overrides: Dict[str, Any] | None = None
    compiled_model_path: str | None = None
    skip_compile: bool = False
    compile_only: bool = False
    compile_dry_run: bool = False
    hlo_debug: bool = False
    on_cpu: bool = False
    enable_torch_dist: bool = True

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        cfg = self.hf_config
        num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        return ModelConfig(
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            vocab_size=cfg.vocab_size,
            max_position=cfg.max_position_embeddings,
        )

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:23333"
