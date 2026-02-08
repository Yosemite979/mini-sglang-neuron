from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import make_positions


@dataclass
class NeuronAttnMetadata(BaseAttnMetadata):
    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.positions[:bs, -1]


class NeuronAttnBackend(BaseAttnBackend):
    def __init__(self, page_table: torch.Tensor) -> None:
        self.page_table = page_table

    def forward(self, q, k, v, layer_id, batch):
        raise RuntimeError("Neuron attention backend is a placeholder and should not be used.")

    def prepare_metadata(self, batch) -> None:
        batch.attn_metadata = NeuronAttnMetadata(positions=make_positions(batch.input_ids.device, batch.padded_reqs))

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        return

    def prepare_for_capture(self, batch) -> None:
        return

    def prepare_for_replay(self, batch) -> None:
        return
