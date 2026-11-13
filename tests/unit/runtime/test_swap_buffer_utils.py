# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import deepspeed.comm as dist
from deepspeed.runtime.swap_tensor.utils import SwapBuffer, SwapBufferPool, SwapBufferManager


class _DummyAioHandle:

    def is_pinned(self, _tensor):
        return True

    def new_cpu_locked_tensor(self, num_elems, example):
        return torch.zeros(num_elems, dtype=example.dtype)

    def free_cpu_locked_tensor(self, _tensor):
        pass


def test_swap_buffer_get_swap_path():
    buf = torch.zeros(16, dtype=torch.float32)
    sb = SwapBuffer(buf)
    sb.allocate_tensor("/tmp/swap_0.tensor", numel=6, aligned_numel=8)
    assert sb.get_swap_path(0) == "/tmp/swap_0.tensor"
    assert sb.get_swap_path(8) is None


def test_swap_buffer_pool_has_space():
    pool = SwapBufferPool([torch.zeros(8), torch.zeros(8), torch.zeros(8)], _DummyAioHandle())
    assert pool.allocate_tensor(4, "/tmp/b0_a", 4)[0] is not None
    assert pool.current_index == 0

    # Requesting more than buffer capacity must not advance current_index past remaining space in buffer 0
    assert not pool.has_space(16)
    assert pool.current_index == 0

    # Remaining capacity in buffer 0 is still usable
    assert pool.allocate_tensor(4, "/tmp/b0_b", 4)[0] is not None
    assert pool.current_index == 0

    # Next allocation advances to buffer 1
    assert pool.allocate_tensor(4, "/tmp/b1_a", 4)[0] is not None
    assert pool.current_index == 1


def test_swap_buffer_manager_zero_count_allocate(monkeypatch):
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    mgr = SwapBufferManager(num_elems=8, count=2, dtype=torch.float32, aio_handle=_DummyAioHandle())
    assert mgr.allocate(num_elems=4, count=0, dtype=torch.float32) == []
    assert len(mgr.free_buffer_index) == 2

    allocated = mgr.allocate_all(num_elems=4, dtype=torch.float32)
    assert len(allocated) == 2

    # Calling allocate_all when no free buffers remain should return [] rather than re-allocating
    assert mgr.allocate_all(num_elems=4, dtype=torch.float32) == []
    mgr.free(allocated)
    assert len(mgr.free_buffer_index) == 2
