# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.models.glm5next.nvidia import attention as attention_module
from vllm.models.glm5next.nvidia import kda as kda_module
from vllm.models.glm5next.nvidia.quantization import (
    Glm5NextAttentionW4A16Config,
    get_glm5_next_attention_quant_config,
)


class _FakeLinear(nn.Module):
    calls: list[tuple[str, object | None]] = []

    def __init__(self, *args, quant_config=None, prefix="", **kwargs):
        super().__init__()
        self.quant_config = quant_config
        self.prefix = prefix
        self.weight = nn.Parameter(torch.empty(1, 1))
        self.register_parameter("bias", None)
        self.calls.append((prefix, quant_config))

    def forward(self, x):
        return x, None


def test_glm5_attention_w4a16_config(monkeypatch):
    checkpoint_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
    )
    monkeypatch.delenv("LVLLM_GLM5_ATTN_W4A16", raising=False)
    assert get_glm5_next_attention_quant_config(checkpoint_config) is None

    monkeypatch.setenv("LVLLM_GLM5_ATTN_W4A16", "1")
    config = get_glm5_next_attention_quant_config(checkpoint_config)
    assert isinstance(config, Glm5NextAttentionW4A16Config)
    assert config.checkpoint_quant_config is checkpoint_config
    assert config.checkpoint_weights_are_fp8

    kda_config = get_glm5_next_attention_quant_config(
        checkpoint_config,
        checkpoint_weights_are_fp8=False,
    )
    assert isinstance(kda_config, Glm5NextAttentionW4A16Config)
    assert not kda_config.checkpoint_weights_are_fp8

    with pytest.raises(RuntimeError, match="requires an FP8 checkpoint"):
        get_glm5_next_attention_quant_config(None)


def test_glm5_kda_w4a16_keeps_recurrent_stack_unquantized(monkeypatch):
    _FakeLinear.calls = []
    monkeypatch.setattr(
        kda_module.GatedDeltaNetAttention,
        "__init__",
        _fake_gdn_init,
    )
    monkeypatch.setattr(
        kda_module,
        "_Glm5NextMergedColumnParallelLinear",
        _FakeLinear,
    )
    monkeypatch.setattr(kda_module, "ColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(kda_module, "RowParallelLinear", _FakeLinear)
    monkeypatch.setattr(
        kda_module,
        "FusedRMSNormGated",
        lambda *args, **kwargs: nn.Identity(),
    )
    monkeypatch.setattr(kda_module, "is_conv_state_dim_first", lambda: True)
    monkeypatch.setattr(
        kda_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        ),
    )

    checkpoint_quant_config = object()
    vllm_config = SimpleNamespace(
        quant_config=checkpoint_quant_config,
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        cache_config=SimpleNamespace(),
        speculative_config=None,
    )
    config = SimpleNamespace(
        hidden_size=4096,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        linear_head_dim=128,
        linear_num_heads=64,
        linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0,
    )

    attention_quant_config = object()
    kda_module.Glm5NextLinearAttention(
        config,
        vllm_config,
        quant_config=attention_quant_config,
        prefix="model.layers.0.self_attn",
    )

    configs = {prefix.rsplit(".", 1)[-1]: quant for prefix, quant in _FakeLinear.calls}
    for name in ("in_proj_qkvbfg_a", "f_b_proj", "g_b_proj", "o_proj"):
        assert configs[name] is attention_quant_config
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        assert configs[name] is None
    assert vllm_config.quant_config is checkpoint_quant_config

def _fake_gdn_init(self, config, vllm_config, prefix):
    nn.Module.__init__(self)
    self.prefix = prefix
    self.tp_size = 1
    self.tp_rank = 0
    self.hidden_size = config.hidden_size
    self.model_config = vllm_config.model_config
    self.cache_config = vllm_config.cache_config
    self.quant_config = vllm_config.quant_config
    self.speculative_config = None
    self.num_spec = 0


def test_glm5_mla_w4a16_quantizes_attention_projections(monkeypatch):
    _FakeLinear.calls = []
    indexer_configs = []
    wrapper_configs = []

    monkeypatch.setattr(
        attention_module,
        "DeepSeekV2FusedQkvAProjLinear",
        _FakeLinear,
    )
    monkeypatch.setattr(attention_module, "ColumnParallelLinear", _FakeLinear)
    monkeypatch.setattr(attention_module, "RowParallelLinear", _FakeLinear)
    monkeypatch.setattr(
        attention_module,
        "RMSNorm",
        lambda *args, **kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        attention_module,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(attention_module, "get_rope", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        attention_module,
        "Indexer",
        lambda *args, **kwargs: indexer_configs.append(args[4]) or nn.Identity(),
    )
    monkeypatch.setattr(attention_module, "MLAModules", lambda **kwargs: kwargs)

    def fake_wrapper(*args, **kwargs):
        wrapper_configs.append(args[10])
        return nn.Identity()

    monkeypatch.setattr(
        attention_module,
        "MultiHeadLatentAttentionWrapper",
        fake_wrapper,
    )

    config = SimpleNamespace(
        rms_norm_eps=1e-5,
        index_topk=2048,
        indexer_rope_interleave=True,
        rope_parameters={"rope_type": "default"},
    )

    attention_quant_config = object()
    attention_module.Glm5NextMLAAttention(
        vllm_config=SimpleNamespace(),
        config=config,
        hidden_size=4096,
        num_heads=64,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        v_head_dim=256,
        q_lora_rank=1536,
        kv_lora_rank=512,
        quant_config=attention_quant_config,
        cache_config=None,
        prefix="model.layers.3.self_attn",
        topk_indices_buffer=None,
        skip_rope=True,
    )

    configs = {prefix.rsplit(".", 1)[-1]: quant for prefix, quant in _FakeLinear.calls}
    for name in ("fused_qkv_a_proj", "q_b_proj", "kv_b_proj", "o_proj"):
        assert configs[name] is attention_quant_config
    assert indexer_configs == [attention_quant_config]
    assert wrapper_configs == [attention_quant_config]
