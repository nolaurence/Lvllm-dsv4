# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.fused_moe.lk_routed_experts import (
    LkRoutedExperts,
    _glm5_lk_shared_expert_count,
)
from vllm.models.glm5next.nvidia.model import (
    Glm5NextDecoderLayer,
    Glm5NextMoE,
    _try_load_glm5_lk_shared_expert,
    _validate_glm5_deferred_moe_allreduce,
    _validate_glm5_shared_expert_cpu,
)


@pytest.mark.parametrize(
    "projection,target,shard",
    [
        ("gate_proj", "w13", "w1"),
        ("down_proj", "w2", "w2"),
        ("up_proj", "w13", "w3"),
    ],
)
@pytest.mark.parametrize("suffix", ["weight", "weight_scale_inv"])
def test_glm5_shared_expert_loader_targets_lk_extra_slot(
    monkeypatch, projection, target, shard, suffix
):
    monkeypatch.setenv("LVLLM_GLM5_SHARED_EXPERT_CPU", "1")
    checkpoint_name = f"model.layers.3.mlp.shared_experts.{projection}.{suffix}"
    parameter_name = f"model.layers.3.mlp.experts.routed_experts.{target}_{suffix}"
    calls = []
    param = nn.Parameter(torch.empty(1))
    param.weight_loader = lambda *args, **kwargs: calls.append((args, kwargs))
    loaded_weight = torch.randn(2, 2)

    mapped_name = _try_load_glm5_lk_shared_expert(
        checkpoint_name,
        loaded_weight,
        {parameter_name: param},
        shared_expert_id=288,
    )

    assert mapped_name == parameter_name
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (param, loaded_weight, parameter_name)
    assert kwargs == {"expert_id": 288, "shard_id": shard}


@pytest.mark.parametrize("weight_mode", ["INT4", "FP8"])
def test_glm5_shared_expert_cpu_validation(monkeypatch, weight_mode):
    monkeypatch.setenv("LVLLM_GLM5_SHARED_EXPERT_CPU", "1")
    monkeypatch.setenv("LVLLM_MOE_NUMA_ENABLED", "1")
    monkeypatch.setenv("LVLLM_MOE_USE_WEIGHT", weight_mode)
    monkeypatch.setenv("LVLLM_ENABLE_MOE_LAYERWISE_LOAD", "1")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    monkeypatch.delenv("LVLLM_GPU_RESIDENT_MOE_LAYERS", raising=False)
    config = SimpleNamespace(n_shared_experts=1)
    parallel = SimpleNamespace(
        enable_expert_parallel=False,
        enable_eplb=False,
        use_sequence_parallel_moe=False,
    )

    assert _validate_glm5_shared_expert_cpu(config, parallel)

    parallel.enable_expert_parallel = True
    with pytest.raises(RuntimeError, match="EP1/EPLB off"):
        _validate_glm5_shared_expert_cpu(config, parallel)
    parallel.enable_expert_parallel = False

    monkeypatch.setenv("LVLLM_GPU_RESIDENT_MOE_LAYERS", "3")
    with pytest.raises(RuntimeError, match="resident GPU MoE"):
        _validate_glm5_shared_expert_cpu(config, parallel)

    monkeypatch.delenv("LVLLM_GPU_RESIDENT_MOE_LAYERS")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "0")
    with pytest.raises(RuntimeError, match="VLLM_USE_BREAKABLE_CUDAGRAPH=1"):
        _validate_glm5_shared_expert_cpu(config, parallel)

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    parallel.use_ubatching = True
    with pytest.raises(RuntimeError, match="ubatching/DBO off"):
        _validate_glm5_shared_expert_cpu(config, parallel)

    parallel.use_ubatching = False
    monkeypatch.delenv("LVLLM_ENABLE_MOE_LAYERWISE_LOAD")
    with pytest.raises(RuntimeError, match="LVLLM_ENABLE_MOE_LAYERWISE_LOAD=1"):
        _validate_glm5_shared_expert_cpu(config, parallel)


def test_glm5_text_config_enables_lk_shared_expert(monkeypatch):
    monkeypatch.setenv("LVLLM_GLM5_SHARED_EXPERT_CPU", "1")
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="glm5_next_text",
                n_shared_experts=1,
            )
        )
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.fused_moe.lk_routed_experts."
        "get_current_vllm_config",
        lambda: config,
    )

    assert _glm5_lk_shared_expert_count() == 1


def test_glm5_deferred_moe_allreduce_validation(monkeypatch):
    monkeypatch.setenv("LVLLM_GLM5_DEFERRED_MOE_ALLREDUCE", "1")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    config = SimpleNamespace(mhc_num_residual_streams=4)
    parallel = SimpleNamespace(
        enable_expert_parallel=False,
        enable_eplb=False,
        use_sequence_parallel_moe=False,
        use_ubatching=False,
    )

    assert _validate_glm5_deferred_moe_allreduce(config, parallel, tp_size=2)
    assert not _validate_glm5_deferred_moe_allreduce(config, parallel, tp_size=1)

    parallel.enable_expert_parallel = True
    with pytest.raises(RuntimeError, match="EP1/EPLB off"):
        _validate_glm5_deferred_moe_allreduce(config, parallel, tp_size=2)


def test_glm5_deferred_moe_allreduce_disables_producer_reduce(monkeypatch):
    import vllm.models.glm5next.nvidia.model as model_module

    class FakeGate(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(288, 8))
            self.out_dtype = torch.float32

    captured = {}
    monkeypatch.setenv("LVLLM_GLM5_DEFERRED_MOE_ALLREDUCE", "1")
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(model_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        model_module,
        "get_ep_group",
        lambda: SimpleNamespace(
            device_group=SimpleNamespace(size=lambda: 2),
            rank_in_group=0,
        ),
    )
    monkeypatch.setattr(
        model_module, "GateLinear", lambda *_args, **_kwargs: FakeGate()
    )
    monkeypatch.setattr(
        model_module,
        "FusedMoEFactory",
        lambda **kwargs: captured.update(kwargs) or nn.Identity(),
    )
    config = SimpleNamespace(
        routed_scaling_factor=2.5,
        n_routed_experts=288,
        n_shared_experts=None,
        hidden_act="silu",
        hidden_size=8,
        topk_method="noaux_tc",
        moe_intermediate_size=4,
        num_experts_per_token=8,
        moe_renormalize=False,
        n_group=8,
        topk_group=4,
        scoring_func="sigmoid",
        swiglu_limit=10.0,
        mhc_num_residual_streams=4,
    )
    parallel = SimpleNamespace(
        use_sequence_parallel_moe=False,
        enable_expert_parallel=False,
        enable_eplb=False,
        eplb_config=SimpleNamespace(num_redundant_experts=0),
        use_ubatching=False,
    )

    moe = Glm5NextMoE(config, parallel, prefix="model.layers.3.mlp")

    assert moe.defer_moe_allreduce
    assert captured["reduce_results"] is False


@pytest.mark.parametrize(
    ("layer_idx", "expected_reduce_inputs", "expected_output"),
    [
        (4, [1.0], 14.0),
        (44, [1.0, 14.0], 24.0),
    ],
)
def test_glm5_deferred_moe_allreduce_happens_at_layer_boundaries(
    monkeypatch,
    layer_idx,
    expected_reduce_inputs,
    expected_output,
):
    class FakeAttention(nn.Module):
        def forward(self, *, hidden_states, positions):
            del positions
            return hidden_states + 1

    class FakeMoE(nn.Module):
        def forward(self, hidden_states, already_sequence_parallel=False):
            assert not already_sequence_parallel
            return hidden_states + 2

    class FakeNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.variance_epsilon = 1e-5

    layer = object.__new__(Glm5NextDecoderLayer)
    nn.Module.__init__(layer)
    layer.mhc = True
    layer.is_mtp_layer = False
    layer.is_sequence_parallel = False
    layer._mlp_is_moe = True
    layer.reduce_previous_moe_output = True
    layer.defer_moe_allreduce = True
    layer.layer_idx = layer_idx
    layer.num_hidden_layers = 45
    layer.n = 4
    layer.self_attn = FakeAttention()
    layer.mlp = FakeMoE()
    layer.input_layernorm = FakeNorm()
    layer.post_attention_layernorm = FakeNorm()
    for name in (
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "hc_ffn_base",
    ):
        setattr(layer, name, nn.Parameter(torch.ones(1)))

    def fake_fused_post_pre(
        self,
        x,
        residual,
        post,
        comb,
        *_args,
        **_kwargs,
    ):
        return residual, post, comb, x

    def fake_hc_post(self, x, residual, post, comb):
        del residual, post, comb
        return x

    layer.hc_fused_post_pre = MethodType(fake_fused_post_pre, layer)
    layer.hc_post = MethodType(fake_hc_post, layer)
    reduce_inputs = []

    def fake_all_reduce(x):
        reduce_inputs.append(float(x.item()))
        return x + 10

    monkeypatch.setattr(
        "vllm.models.glm5next.nvidia.model.tensor_model_parallel_all_reduce",
        fake_all_reduce,
    )
    monkeypatch.setattr(
        "vllm.models.glm5next.nvidia.model.hc_contract",
        lambda x, _n: x,
    )
    value = torch.tensor([[1.0]])

    output, *_ = layer(
        positions=torch.zeros(1, dtype=torch.long),
        hidden_states=value,
        residual=value,
        post=value,
        comb=value,
    )

    assert reduce_inputs == expected_reduce_inputs
    torch.testing.assert_close(output, torch.tensor([[expected_output]]))


def test_glm5_moe_gate_is_owned_by_runner():
    class UnexpectedGate(nn.Module):
        def forward(self, _):
            raise AssertionError("GLM model must not launch the gate twice")

    class FakeExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.router_logits = None

        def forward(self, *, hidden_states, router_logits):
            self.router_logits = router_logits
            return hidden_states + 1

    moe = object.__new__(Glm5NextMoE)
    nn.Module.__init__(moe)
    moe.is_sequence_parallel = False
    moe.gate = UnexpectedGate()
    moe.experts = FakeExperts()
    hidden_states = torch.randn(2, 8)

    output = moe(hidden_states)

    assert moe.experts.router_logits is hidden_states
    torch.testing.assert_close(output, hidden_states + 1)


def test_lk_piecewise_entry_writes_stable_output(monkeypatch):
    layer = object.__new__(LkRoutedExperts)
    nn.Module.__init__(layer)
    layer.local_num_experts = 1
    layer.lk_extra_shared_experts = 0
    layer.layer_name = "test"
    hidden_states = torch.randn(2, 8)
    weights = torch.ones(2, 1)
    ids = torch.zeros(2, 1, dtype=torch.int32)
    output = torch.empty_like(hidden_states)
    expected = hidden_states + 3

    def fake_decode(self, hidden, topk_weights, mapped_ids, output=None):
        assert hidden is hidden_states
        assert topk_weights is weights
        assert mapped_ids is ids
        return expected.clone()

    monkeypatch.setattr(
        layer,
        "_forward_lk_cuda_decode",
        MethodType(fake_decode, layer),
    )

    actual = layer._forward_lk_cuda_decode_into(hidden_states, weights, ids, output)

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected)


def test_kt_fp8_piecewise_entry_writes_stable_output(monkeypatch):
    buffers = {}

    class FakeMoe:
        def forward_task(self, _bsz, k, _ids, _weights, _input, _output, incremental):
            assert k == 9
            assert not incremental

            def run():
                buffers[("kt_fp8_output", (2, 8), torch.float32)].copy_(
                    buffers[("kt_fp8_input", (2, 8), torch.float32)] + 5
                )

            return run

    class FakeCpuInfer:
        def submit(self, task):
            task()

        def sync(self):
            pass

    layer = object.__new__(LkRoutedExperts)
    nn.Module.__init__(layer)
    layer.lk_moe = FakeMoe()
    layer._kt_fp8_max_len = 4
    layer.local_num_experts = 1
    layer.lk_extra_shared_experts = 0
    layer.layer_name = "test"

    def get_buffer(self, name, shape, dtype):
        key = (name, shape, dtype)
        return buffers.setdefault(key, torch.empty(shape, dtype=dtype))

    monkeypatch.setattr(layer, "_get_lk_cpu_buffer", MethodType(get_buffer, layer))
    monkeypatch.setattr(layer, "_get_kt_fp8_runtime", lambda: (None, FakeCpuInfer()))
    hidden_states = torch.randn(2, 8)
    weights = torch.ones(2, 9)
    ids = torch.zeros(2, 9, dtype=torch.int64)
    output = torch.empty_like(hidden_states)

    actual = layer._forward_kt_fp8_into(hidden_states, weights, ids, output)

    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, hidden_states + 5)


def test_kt_fp8_piecewise_entry_chunks_prefill(monkeypatch):
    buffers = {}
    chunk_sizes = []
    cursor = 0

    class FakeMoe:
        def forward_task(self, _bsz, _k, _ids, _weights, _input, _output, _inc):
            nonlocal cursor
            start = cursor

            def run():
                nonlocal cursor
                chunk_size = int(buffers[("kt_fp8_bsz", (1,), torch.int32)][0])
                chunk_sizes.append(chunk_size)
                output = buffers[("kt_fp8_output", (5, 8), torch.float32)]
                hidden = buffers[("kt_fp8_input", (5, 8), torch.float32)]
                output[start : start + chunk_size].copy_(
                    hidden[start : start + chunk_size] + 7
                )
                cursor += chunk_size

            return run

    class FakeCpuInfer:
        def submit(self, task):
            task()

        def sync(self):
            pass

    layer = object.__new__(LkRoutedExperts)
    nn.Module.__init__(layer)
    layer.lk_moe = FakeMoe()
    layer._kt_fp8_max_len = 2
    layer.local_num_experts = 1
    layer.lk_extra_shared_experts = 0
    layer.layer_name = "test"

    def get_buffer(self, name, shape, dtype):
        key = (name, shape, dtype)
        return buffers.setdefault(key, torch.empty(shape, dtype=dtype))

    monkeypatch.setattr(layer, "_get_lk_cpu_buffer", MethodType(get_buffer, layer))
    monkeypatch.setattr(layer, "_get_kt_fp8_runtime", lambda: (None, FakeCpuInfer()))
    hidden_states = torch.randn(5, 8)
    weights = torch.ones(5, 9)
    ids = torch.zeros(5, 9, dtype=torch.int64)
    output = torch.empty_like(hidden_states)

    actual = layer._forward_kt_fp8_into(hidden_states, weights, ids, output)

    assert chunk_sizes == [2, 2, 1]
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, hidden_states + 7)


def test_lk_cpu_fallback_includes_shared_expert_in_topk(monkeypatch):
    class FakeLkMoe:
        def __init__(self):
            self.k = None

        def forward(self, _qlen, k, *_args):
            self.k = k

    layer = object.__new__(LkRoutedExperts)
    nn.Module.__init__(layer)
    layer.lk_moe = FakeLkMoe()
    layer.lk_extra_shared_experts = 1
    layer.local_num_experts = 288
    layer._expert_map = None
    layer.layer_name = "model.layers.3.mlp.experts"
    buffers = {}

    def get_buffer(self, name, shape, dtype):
        key = (name, shape, dtype)
        return buffers.setdefault(key, torch.empty(shape, dtype=dtype))

    monkeypatch.setattr(
        layer,
        "_get_lk_cpu_buffer",
        MethodType(get_buffer, layer),
    )
    hidden_states = torch.randn(2, 8)
    weights = torch.full((2, 8), 0.125)
    ids = torch.arange(8, dtype=torch.int32).repeat(2, 1)

    layer.forward_lk(hidden_states, weights, ids)

    assert layer.lk_moe.k == 9
    expert_ids = buffers[("expert_ids", (2, 9), torch.uint64)]
    assert torch.equal(expert_ids[:, -1], torch.full((2,), 288, dtype=torch.uint64))
