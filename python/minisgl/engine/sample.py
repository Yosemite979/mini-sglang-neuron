from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    temperatures: torch.Tensor | None
    top_k: torch.Tensor | None = None
    top_p: torch.Tensor | None = None


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    #return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)
    return torch.tensor(data, dtype=dtype).to(device, non_blocking=True)


@dataclass
class Sampler:
    device: torch.device
    vocab_size: int

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [1.0 if p.is_greedy else max(p.temperature, MIN_T) for p in params]
        # Encode per-request greedy mode as top_k=1 for mixed batches.
        top_ks = [1 if p.is_greedy else (p.top_k if p.top_k >= 1 else self.vocab_size) for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        # NxDI send logits to CPU by default.
        # when args.temperatures is None all requests are greedy: simply take argmax
        if args.temperatures is None:
            # logits shape: (N, vocab_size)
            # return int32 tensor of selected indices
            return torch.argmax(logits, dim=-1).to(dtype=torch.int32)
        return self._sample_cpu(logits, args)

    def _sample_cpu(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        assert args.temperatures is not None
        logits = logits.float()
        cpu_device = logits.device
        temperatures = args.temperatures.to(device=cpu_device)
        scaled_logits = logits / temperatures.unsqueeze(1)

        batch_size, vocab_size = scaled_logits.shape
        output = torch.empty((batch_size,), dtype=torch.int32, device=cpu_device)

        top_k = (
            args.top_k.to(device=cpu_device, dtype=torch.int64)
            if args.top_k is not None
            else torch.full((batch_size,), vocab_size, dtype=torch.int64, device=cpu_device)
        )
        top_p = (
            args.top_p.to(device=cpu_device)
            if args.top_p is not None
            else torch.ones((batch_size,), dtype=torch.float32, device=cpu_device)
        )

        topk_rows = (top_k > 0) & (top_k < vocab_size)
        if topk_rows.any():
            topk_indices = torch.nonzero(topk_rows, as_tuple=False).squeeze(1)
            row_logits = scaled_logits[topk_indices]
            row_top_k = top_k[topk_indices]
            max_k = int(row_top_k.max().item())

            topk_values, topk_tokens = torch.topk(row_logits, k=max_k, dim=-1)
            valid = torch.arange(max_k, device=cpu_device).unsqueeze(0) < row_top_k.unsqueeze(1)
            masked_values = topk_values.masked_fill(~valid, float("-inf"))
            probs = torch.softmax(masked_values, dim=-1)
            sampled = torch.multinomial(probs, 1)
            output[topk_indices] = torch.gather(topk_tokens, 1, sampled).squeeze(1).to(torch.int32)

        topp_rows = (~topk_rows) & (top_p < 1.0)
        if topp_rows.any():
            topp_indices = torch.nonzero(topp_rows, as_tuple=False).squeeze(1)
            row_logits = scaled_logits[topp_indices]
            row_top_p = top_p[topp_indices]

            probs = torch.softmax(row_logits, dim=-1)
            sorted_probs, sorted_tokens = torch.sort(probs, dim=-1, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            valid = cumulative <= row_top_p.unsqueeze(1)
            valid[:, 0] = True

            filtered_probs = sorted_probs.masked_fill(~valid, 0.0)
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
            sampled = torch.multinomial(filtered_probs, 1)
            output[topp_indices] = torch.gather(sorted_tokens, 1, sampled).squeeze(1).to(torch.int32)

        greedy_rows = ~(topk_rows | topp_rows)
        if greedy_rows.any():
            greedy_indices = torch.nonzero(greedy_rows, as_tuple=False).squeeze(1)
            output[greedy_indices] = torch.argmax(scaled_logits[greedy_indices], dim=-1).to(torch.int32)

        return output
