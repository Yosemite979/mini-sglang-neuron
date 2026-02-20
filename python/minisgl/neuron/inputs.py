from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List

import torch

from minisgl.core import Batch
from minisgl.utils import init_logger

logger = init_logger(__name__)

_SLOT_MAPPING_PAD = -1
_BLOCK_TABLE_PAD = 0

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
        device_lens = [req.device_len for req in reqs]
        max_device_len = max(device_lens) if device_lens else 0

        input_block_ids = torch.empty((batch_size,), dtype=torch.int32, device=batch.input_ids.device)

        #logger.error(f"xinux - extend_lens={extend_lens}")
        #logger.error(f"xinux - max_extend_len={max_extend_len}")
        #logger.error(f"xinux - {batch.is_prefill=}")
        #logger.error(f"xinux - {reqs[0]=}")
        #logger.error(f"xinux - {self.page_table[0].tolist()[:100]=}")
        
        if batch.is_prefill:
            input_tokens = torch.full(
                (batch_size, max_device_len),
                self.pad_token_id,
                dtype=torch.int32,
                device=batch.input_ids.device,
            )
            position_ids = torch.zeros(
                (batch_size, max_device_len),
                dtype=torch.int32,
                device=batch.input_ids.device,
            )
            slot_mapping = torch.full(
                (batch_size, self.max_seq_len),
                _SLOT_MAPPING_PAD,
                dtype=torch.int32,
                device=batch.input_ids.device,
            )
            block_tables = torch.full(
                (len(reqs), self.max_seq_len),
                _BLOCK_TABLE_PAD,
                dtype=torch.int32,
                device=batch.input_ids.device,
            )
        else:
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
                _SLOT_MAPPING_PAD,
                dtype=torch.int32,
                device=batch.input_ids.device,
            )
            block_tables = torch.full(
                (len(reqs), self.max_seq_len),
                _BLOCK_TABLE_PAD,
                dtype=torch.int32,
                device=batch.input_ids.device,
            )

        

        # Build per-request inputs from the concatenated batch.input_ids.
        offset = 0
        for i, req in enumerate(reqs):
            ext_len = req.extend_len
            dev_len = req.device_len
            cached_len = req.cached_len
            if ext_len > 0:
                if batch.is_prefill:
                    tokens = batch.input_ids[offset : offset + dev_len]
                    input_tokens[i, :dev_len] = tokens
                    position_ids[i, :dev_len] = torch.arange(
                        0,
                        dev_len,
                        dtype=torch.int32,
                        device=batch.input_ids.device,
                    )
                    block_tables[i, : dev_len] = self._build_block_tables(req)
                    slot_mapping[i, :(dev_len-cached_len)] = block_tables[i, cached_len:dev_len]
                    offset += dev_len
                else:
                    tokens = batch.input_ids[offset : offset + ext_len]
                    input_tokens[i, :ext_len] = tokens
                    position_ids[i, :ext_len] = torch.arange(
                        req.cached_len-1,
                        req.cached_len-1 + ext_len,
                        dtype=torch.int32,
                        device=batch.input_ids.device,
                    )
                    block_tables[i, :dev_len] = self._build_block_tables(req)
                    slot_mapping[i, :ext_len] = block_tables[i, req.cached_len-1:req.cached_len-1+ext_len] 
                    offset += ext_len 
            input_block_ids[i] = req.table_idx

        #block_tables = self._build_block_tables(reqs, batch.input_ids.device, is_prefill=batch.is_prefill)

        if batch.is_prefill:
            full_context_lens = [req.device_len for req in reqs]
            computed_context_lens = [req.cached_len for req in reqs]
        else:
            full_context_lens = [req.cached_len for req in reqs]
            computed_context_lens = [req.cached_len-1 for req in reqs]

        full_context_lens=torch.tensor(
            full_context_lens, dtype=torch.int32, device=batch.input_ids.device
        ).reshape(-1, 1)

        computed_context_lens=torch.tensor(
            computed_context_lens, dtype=torch.int32, device=batch.input_ids.device
        ).reshape(-1, 1)

        return ModelInputForNeuron(
            input_tokens=input_tokens,
            position_ids=position_ids,
            input_block_ids=input_block_ids,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            full_context_lens=full_context_lens,
            computed_context_lens=computed_context_lens,
        )

    def _build_block_tables(self, req) -> torch.Tensor:
        return copy.deepcopy(self.page_table[req.table_idx, : req.device_len])
