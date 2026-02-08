from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from minisgl.core import Batch


@dataclass(frozen=True)
class ModelInputForNeuron:
    input_tokens: torch.Tensor
    position_ids: torch.Tensor
    input_block_ids: torch.Tensor
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    full_context_lens: torch.Tensor
    computed_context_lens: torch.Tensor


class NeuronInputBuilder:
    def __init__(self, page_table: torch.Tensor, max_seq_len: int, pad_token_id: int = 0):
        self.page_table = page_table
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def build(self, batch: Batch) -> ModelInputForNeuron:
        reqs = batch.padded_reqs
        batch_size = len(reqs)

        extend_lens = [req.extend_len for req in reqs]
        max_extend_len = max(extend_lens) if extend_lens else 0
        full_context_lens = [req.device_len for req in reqs]
        computed_context_lens = [req.cached_len for req in reqs]

        input_tokens = torch.full(
            (batch_size, max_extend_len),
            self.pad_token_id,
            dtype=torch.int32,
            device=batch.input_ids.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_extend_len),
            dtype=torch.int32,
            device=batch.input_ids.device,
        )
        slot_mapping = torch.full(
            (batch_size, max_extend_len),
            -1,
            dtype=torch.int32,
            device=batch.input_ids.device,
        )
        input_block_ids = torch.empty((batch_size,), dtype=torch.int32, device=batch.input_ids.device)

        # Build per-request inputs from the concatenated batch.input_ids.
        offset = 0
        for i, req in enumerate(reqs):
            ext_len = req.extend_len
            if ext_len > 0:
                tokens = batch.input_ids[offset : offset + ext_len]
                input_tokens[i, :ext_len] = tokens
                position_ids[i, :ext_len] = torch.arange(
                    req.cached_len,
                    req.cached_len + ext_len,
                    dtype=torch.int32,
                    device=batch.input_ids.device,
                )
                slot_mapping[i, :ext_len] = self.page_table[
                    req.table_idx, req.cached_len : req.cached_len + ext_len
                ]
                offset += ext_len
            input_block_ids[i] = req.table_idx

        block_tables = self._build_block_tables(reqs, batch.input_ids.device)

        return ModelInputForNeuron(
            input_tokens=input_tokens,
            position_ids=position_ids,
            input_block_ids=input_block_ids,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            full_context_lens=torch.tensor(
                full_context_lens, dtype=torch.int32, device=batch.input_ids.device
            ),
            computed_context_lens=torch.tensor(
                computed_context_lens, dtype=torch.int32, device=batch.input_ids.device
            ),
        )

    def _build_block_tables(self, reqs: List, device: torch.device) -> torch.Tensor:
        block_tables = torch.full(
            (len(reqs), self.max_seq_len),
            0,
            dtype=torch.int32,
            device=device,
        )
        for i, req in enumerate(reqs):
            if req.device_len > 0:
                block_tables[i, : req.device_len] = self.page_table[
                    req.table_idx, : req.device_len
                ]
        return block_tables
