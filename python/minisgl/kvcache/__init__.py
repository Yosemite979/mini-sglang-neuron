from __future__ import annotations

from typing import Protocol

from minisgl.utils import Registry

import torch

from .base import (
    BaseCacheHandle,
    BaseCacheManager,
    SizeInfo,
)


class CacheManagerCreator(Protocol):
    def __call__(self, device: torch.device) -> BaseCacheManager: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache_manager(device: torch.device):
    from .naive_manager import NaiveCacheManager

    return NaiveCacheManager(device=device)


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache_manager(device: torch.device):
    from .radix_manager import RadixCacheManager

    return RadixCacheManager(device=device)


def create_cache_manager(device: torch.device, type: str) -> BaseCacheManager:
    return SUPPORTED_CACHE_MANAGER[type](device)


__all__ = [
    "create_cache_manager",
    "BaseCacheHandle",
    "BaseCacheManager",
    "SizeInfo",
    "SUPPORTED_CACHE_MANAGER",
]
