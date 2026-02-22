import torch


class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs
        self._free_slots = list(range(max_running_reqs))
        self.page_table = page_table
        # The last row of the page_table is reserved for dummy requests, which should never be allocated to real requests. This simplifies the handling of padded dummy requests.

        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)
        # Similarly, the last slot of the token_pool is reserved for dummy requests, which should be initialized with pad_token_id (e.g., 0) and never overwritten by real requests.
        # It simplifies the handling of padded dummy requests and chunked requests that require padding.

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)
