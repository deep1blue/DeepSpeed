# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
from torch._utils import _flatten_dense_tensors
import deepspeed.comm as dist
import pytest
from typing import Dict

import deepspeed.runtime.utils as ds_utils
import deepspeed.utils.groups as groups
from deepspeed.accelerator import get_accelerator

from unit.common import DistributedTest


def test_call_to_str():
    c2s = ds_utils.call_to_str

    assert c2s('int') == 'int()'
    assert c2s('int', 3) == 'int(3)'
    assert c2s('int', 3, 'jeff') == 'int(3, \'jeff\')'

    assert c2s('hello', val=3) == 'hello(val=3)'
    assert c2s('hello', 1138, val=3) == 'hello(1138, val=3)'


class TestClipGradNorm(DistributedTest):
    world_size = 2

    def test_gather(self):
        param1 = torch.nn.Parameter(torch.Tensor([0]))
        param1.grad = torch.Tensor([1])
        param2 = torch.nn.Parameter(torch.Tensor([0]))
        param2.grad = torch.Tensor([dist.get_rank() + 1])
        # param2 is now MoE parameter
        param2.allreduce = False

        parameters = [param1, param2]

        groups._create_expert_and_data_parallel(2)

        norm = ds_utils.clip_grad_norm_(parameters, max_norm=0.1)
        norm = torch.Tensor([norm]).to(get_accelerator().device_name(dist.get_rank()))
        world_size = dist.get_world_size()
        gathered_norm = [torch.zeros(1).to(get_accelerator().device_name()) for i in range(world_size)]

        dist.all_gather(gathered_norm, norm)

        assert gathered_norm[0] == gathered_norm[1], "norm at rank 0 does not match the norm at rank 1"

    def test_clipped_val(self):
        max_norm = 0.1

        def test_params():
            param1 = torch.nn.Parameter(torch.Tensor([0]))
            param1.grad = torch.Tensor([1])
            param2 = torch.nn.Parameter(torch.Tensor([0]))
            param2.grad = torch.Tensor([1])
            return [param1, param2]

        # This assumes gradients are same on all the ranks and doesn't consider multiple ranks
        params_expected = test_params()
        torch.nn.utils.clip_grad_norm_(params_expected, max_norm)

        params_actual = test_params()
        ds_utils.clip_grad_norm_(params_actual, max_norm=max_norm)

        # This can be allclose
        assert torch.equal(params_expected[0].grad, params_actual[0].grad)
        assert torch.equal(params_expected[1].grad, params_actual[1].grad)


class TestClipGradNormPNorm(DistributedTest):
    # world_size 1 so this runs wherever the suite runs; the bug is in the per-rank
    # recombination of the norms, which is independent of the group size.
    world_size = 1

    @pytest.mark.parametrize("norm_type", [1, 2, 3])
    def test_matches_torch(self, norm_type):
        # The p-norm over all gradients is (sum_i ||g_i||_p ** p) ** (1/p). Squaring the
        # per-parameter norms computes that only for p == 2, which is the control here.
        def test_params():
            param1 = torch.nn.Parameter(torch.zeros(2))
            param1.grad = torch.Tensor([3.0, -4.0])
            param2 = torch.nn.Parameter(torch.zeros(1))
            param2.grad = torch.Tensor([2.0])
            return [param1, param2]

        max_norm = 1.0
        params_expected = test_params()
        expected_norm = torch.nn.utils.clip_grad_norm_(params_expected, max_norm, norm_type=norm_type)

        params_actual = test_params()
        actual_norm = ds_utils.clip_grad_norm_(params_actual, max_norm=max_norm, norm_type=norm_type)

        assert torch.allclose(actual_norm.float().cpu(), expected_norm.float().cpu())
        for expected, actual in zip(params_expected, params_actual):
            assert torch.allclose(actual.grad, expected.grad)


@pytest.mark.parametrize("check_using_norm", [(False), (True)])
class TestCheckOverflow(DistributedTest):
    world_size = 2

    def test(self, check_using_norm):
        groups._create_expert_and_data_parallel(2)

        param1 = torch.nn.Parameter(torch.Tensor([0]))
        param1.grad = torch.Tensor([1])
        param2 = torch.nn.Parameter(torch.Tensor([0]))
        if dist.get_rank() == 0:
            param2.grad = torch.Tensor([1])
        else:
            param2.grad = torch.Tensor([float("inf")])
        param2.allreduce = False
        # param2 is now MoE parameter
        parameters = [param1, param2]
        if check_using_norm:
            grads_group_flat = [_flatten_dense_tensors([p.grad for p in parameters])]
            norm = ds_utils.get_weight_norm(grads_group_flat)
            overflow_checker = ds_utils.CheckOverflow([parameters])
            overflow = overflow_checker.check_using_norm([norm], reduce_overflow=False)
        else:
            overflow_checker = ds_utils.CheckOverflow([parameters])
            overflow = overflow_checker.check()
        assert overflow


@pytest.mark.skipif(not hasattr(torch.autograd.graph, "_get_grad_fn_or_grad_acc"),
                    reason="requires torch.autograd.graph._get_grad_fn_or_grad_acc")
def test_count_used_parameters_enables_grad_for_grad_acc_lookup(monkeypatch):
    """count_used_parameters_in_backward should enable grad for grad-acc lookup."""
    param = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
    seen: Dict[str, int] = {"lookup_calls": 0}
    original_getter = torch.autograd.graph._get_grad_fn_or_grad_acc

    def _require_grad_enabled(t):
        seen["lookup_calls"] += 1
        if not torch.is_grad_enabled():
            raise RuntimeError("grad mode must be enabled for grad-acc lookup")
        return original_getter(t)

    monkeypatch.setattr(torch.autograd.graph, "_get_grad_fn_or_grad_acc", _require_grad_enabled)

    def _hook(grad):
        seen["count"] = ds_utils.count_used_parameters_in_backward([param])
        return grad

    param.register_hook(_hook)
    loss = (param * 2.0).sum()
    loss.backward()
    assert seen["lookup_calls"] > 0
    assert "count" in seen


def test_swap_buffer_get_swap_path_and_pool_has_space(monkeypatch):
    from deepspeed.runtime.swap_tensor.utils import SwapBuffer, SwapBufferPool, SwapBufferManager

    buf = torch.zeros(16, dtype=torch.float32)
    sb = SwapBuffer(buf)
    sb.allocate_tensor("/tmp/swap_0.tensor", numel=6, aligned_numel=8)
    assert sb.get_swap_path(0) == "/tmp/swap_0.tensor"
    assert sb.get_swap_path(8) is None

    class _DummyAioHandle:
        def is_pinned(self, _tensor):
            return True

        def new_cpu_locked_tensor(self, num_elems, example):
            return torch.zeros(num_elems, dtype=example.dtype)

        def free_cpu_locked_tensor(self, _tensor):
            pass

    pool = SwapBufferPool([torch.zeros(8), torch.zeros(8), torch.zeros(8)], _DummyAioHandle())
    # Fill buffer 0 and buffer 1
    assert pool.allocate_tensor(8, "/tmp/b0", 8)[0] is not None
    assert pool.allocate_tensor(8, "/tmp/b1", 8)[0] is not None
    assert pool.current_index == 1

    # Requesting more than buffer capacity must not advance current_index to an unused buffer
    assert not pool.has_space(16)
    assert pool.current_index == 1

    # Requesting a tensor that fits in buffer 2 advances across full buffers
    assert pool.allocate_tensor(4, "/tmp/b2", 4)[0] is not None
    assert pool.current_index == 2

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

