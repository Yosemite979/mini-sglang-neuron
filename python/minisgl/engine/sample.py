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
        output = torch.empty((logits.shape[0],), dtype=torch.int32)
        temperatures = args.temperatures
        top_k = args.top_k if args.top_k is not None else None
        top_p = args.top_p if args.top_p is not None else None

        for i in range(logits.shape[0]):
            row = logits[i].float()
            temperature = float(temperatures[i].item())
            if temperature > 0:
                row = row / temperature

            k = int(top_k[i].item()) if top_k is not None else self.vocab_size
            p = float(top_p[i].item()) if top_p is not None else 1.0

            if k > 0 and k < row.numel():
                topk = torch.topk(row, k=k)
                probs = torch.softmax(topk.values, dim=-1)
                idx = torch.multinomial(probs, 1)
                token = topk.indices[idx]
            elif p < 1.0:
                probs = torch.softmax(row, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative <= p
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
