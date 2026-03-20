from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.kvcache import BaseCacheHandle, create_cache_manager

if TYPE_CHECKING:
    from .utils import PendingReq


class CacheManager:
    def __init__(self, device: torch.device, num_pages: int, type: str):
        # TODO: support page_size > 1
        # Page ID 0 is reserved for dummy padding, so real cache pages are 1..num_pages.
        self._free_slots = torch.arange(1, num_pages + 1, dtype=torch.int32, device=device)
        self._free_len = num_pages
        self.device = device
        self.manager = create_cache_manager(device=device, type=type)
        self.num_pages = num_pages

    def _free(self, indices: torch.Tensor) -> None:
        free_len = len(indices)
        if free_len == 0:
            return
        next_free_len = self._free_len + free_len
        assert next_free_len <= self.num_pages, "Free slot pool overflow."
        self._free_slots[self._free_len : next_free_len].copy_(indices)
        self._free_len = next_free_len

    def _allocate_from_free(self, needed_len: int) -> torch.Tensor:
        assert needed_len <= self._free_len
        start = self._free_len - needed_len
        allocated = self._free_slots[start : self._free_len].clone()
        self._free_len = start
        return allocated

    def match_req(self, req: PendingReq):
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.manager.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        return self.manager.size_info.evictable_size + self._free_len

    def lock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        self.manager.lock_handle(handle, unlock=True)

    def allocate(self, needed_len: int) -> torch.Tensor:
        if needed_len <= self._free_len:
            return self._allocate_from_free(needed_len)

        # NOTE: len(evicted) + free_len >= needed_len
        evicted = self.manager.evict(needed_len - self._free_len) 
        self._free(evicted)
        assert self._free_len >= needed_len, "Eviction did not free enough space."
        return self._allocate_from_free(needed_len)

    def free_and_cache_finished_req(
        self,
        old_handle: BaseCacheHandle,
        input_ids: torch.Tensor,
        indices: torch.Tensor,
    ) -> None:
        in_cache_len = self.manager.insert_prefix(input_ids, indices)
        self._free(indices[old_handle.cached_len : in_cache_len])
        self.unlock(old_handle)

    def check_integrity(self) -> None:
        self.manager.check_integrity()
        if self._free_len + self.manager.size_info.total_size != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_slots({self._free_len}) +"
                f" total_size({self.manager.size_info.total_size}) != num_pages({self.num_pages})"
            )
