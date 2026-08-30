# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the Triton sparse MLA kernel.

Compares split-KV against the single-pass (`num_kv_splits=1`) path
produced by the same kernel — both paths must agree to within bf16 ULPs.
"""

from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    _DIM_QK,
    _DIM_QK_NOPE,
    _SUPPORTED_DIM_QK,
    triton_mla_sparse_attention,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton sparse MLA kernel requires CUDA/ROCm",
)


@pytest.fixture(scope="module")
def kv_cache():
    torch.manual_seed(0)
    return torch.randn(32768, 1, _DIM_QK, dtype=torch.bfloat16, device="cuda")


def _assert_split_matches_single_pass(
    num_tokens: int,
    num_heads: int,
    topk: int,
    num_kv_splits: int | None,
    kv_cache: torch.Tensor,
) -> None:
    torch.manual_seed(0)
    q = torch.randn(num_tokens, num_heads, _DIM_QK, dtype=torch.bfloat16, device="cuda")
    indices = torch.randint(
        0, kv_cache.shape[0], (num_tokens, 1, topk), dtype=torch.int32, device="cuda"
    )
    out_ref = triton_mla_sparse_attention(
        q,
        kv_cache,
        indices,
        sm_scale=0.1,
        num_kv_splits=1,
    )
    out = triton_mla_sparse_attention(
        q,
        kv_cache,
        indices,
        sm_scale=0.1,
        num_kv_splits=num_kv_splits,
    )
    torch.testing.assert_close(
        out.float(),
        out_ref.float(),
        atol=5e-2,
        rtol=5e-3,
    )


@pytest.mark.parametrize(
    "num_tokens,num_heads",
    [(1, 16), (1, 128), (8, 32), (32, 128), (128, 16)],
)
@pytest.mark.parametrize("topk", [1024, 2048, 4096])
@pytest.mark.parametrize("num_kv_splits", [2, 4, 8])
def test_split_kv_matches_single_pass(
    num_tokens, num_heads, topk, num_kv_splits, kv_cache
):
    _assert_split_matches_single_pass(
        num_tokens,
        num_heads,
        topk,
        num_kv_splits,
        kv_cache,
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 32, 128])
def test_auto_split_matches_single_pass(num_tokens, kv_cache):
    _assert_split_matches_single_pass(
        num_tokens,
        num_heads=128,
        topk=2048,
        num_kv_splits=None,
        kv_cache=kv_cache,
    )


@pytest.mark.parametrize("num_kv_splits", [1, 2, 4, 8])
def test_short_prefill_no_nan(num_kv_splits, kv_cache):
    """Regression: short prefill where most topk slots are -1 sentinels.

    The indexer fills 2048 topk positions with only a handful of valid
    indices; the rest are -1. Before the NEG_LARGE sentinel fix, the online
    softmax produced NaN via `max(-inf, -inf) = -inf` and
    `exp2(-inf − -inf) = NaN`, poisoning every split.
    """
    torch.manual_seed(0)
    num_tokens, num_heads, topk = 5, 16, 2048
    q = torch.randn(num_tokens, num_heads, _DIM_QK, dtype=torch.bfloat16, device="cuda")
    indices = torch.full((num_tokens, 1, topk), -1, dtype=torch.int32, device="cuda")
    # Only the first `t+1` slots of each query hold valid indices; the
    # remaining ~2045 slots are -1, producing many all-invalid BLOCK_N tiles.
    for t in range(num_tokens):
        indices[t, 0, : t + 1] = torch.arange(
            64, 64 + t + 1, dtype=torch.int32, device="cuda"
        )
    out = triton_mla_sparse_attention(
        q, kv_cache, indices, sm_scale=0.0417, num_kv_splits=num_kv_splits
    )
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.parametrize("head_size", _SUPPORTED_DIM_QK)
@pytest.mark.parametrize("num_kv_splits", [1, 2])
def test_supported_head_sizes_match_torch_reference(head_size, num_kv_splits):
    torch.manual_seed(1)
    num_tokens, num_heads, seq_kv, topk = 2, 8, 128, 64
    sm_scale = 0.1
    q = torch.randn(
        num_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda"
    )
    kv = torch.randn(seq_kv, 1, head_size, dtype=torch.bfloat16, device="cuda")
    indices = torch.randint(
        0, seq_kv, (num_tokens, 1, topk), dtype=torch.int32, device="cuda"
    )

    out = triton_mla_sparse_attention(
        q,
        kv,
        indices,
        sm_scale=sm_scale,
        num_kv_splits=num_kv_splits,
    )

    selected_kv = kv[indices[:, 0], 0].float()
    scores = torch.einsum("thd,tkd->thk", q.float(), selected_kv) * sm_scale
    probs = torch.softmax(scores, dim=-1)
    expected = torch.einsum(
        "thk,tkd->thd", probs, selected_kv[..., :_DIM_QK_NOPE]
    ).to(torch.bfloat16)
    torch.testing.assert_close(out, expected, atol=5e-2, rtol=5e-3)


def test_backend_supports_nope_and_rope_head_sizes():
    from vllm.v1.attention.backends.mla.triton_mla_sparse import (
        TritonMLASparseBackend,
    )

    assert TritonMLASparseBackend.get_supported_head_sizes() == list(
        _SUPPORTED_DIM_QK
    )


def test_warmup_uses_runtime_head_size(monkeypatch):
    import vllm.v1.attention.backends.mla.triton_mla_sparse as backend_module

    impl = object.__new__(backend_module.TritonMLASparseImpl)
    impl.topk_indices_buffer = torch.empty(1, 16, dtype=torch.int32, device="cuda")
    impl.num_heads = 8
    impl.head_size = _DIM_QK_NOPE
    impl.softmax_scale = 0.1
    impl._sm_count = 1
    seen_head_sizes = []

    def fake_sparse_attention(q, kv, *args, **kwargs):
        seen_head_sizes.append((q.shape[-1], kv.shape[-1]))
        return torch.empty(
            q.shape[0], q.shape[1], _DIM_QK_NOPE, dtype=q.dtype, device=q.device
        )

    monkeypatch.setattr(
        backend_module, "triton_mla_sparse_attention", fake_sparse_attention
    )
    monkeypatch.setattr(
        backend_module, "warmup_fp8_mqa_logits_triton", lambda **_: None
    )
    monkeypatch.setattr(
        backend_module, "get_current_vllm_config_or_none", lambda: None
    )

    impl._warmup_autotune(SimpleNamespace(n_head=8, head_dim=128))

    assert seen_head_sizes == [
        (_DIM_QK_NOPE, _DIM_QK_NOPE)
    ] * len(backend_module.KV_SPLITS_CANDIDATES)


def test_backend_remaps_padded_kpool_buffer_width(monkeypatch):
    import vllm.v1.attention.backends.mla.xpu_mla_sparse as base_module
    from vllm.v1.attention.backends.mla.triton_mla_sparse import (
        TritonMLASparseImpl,
    )

    num_tokens = 2
    buffer_width = 2176
    impl = object.__new__(TritonMLASparseImpl)
    impl.num_heads = 8
    impl.kv_cache_dtype = "bfloat16"
    impl.topk_indices_buffer = torch.full(
        (num_tokens, buffer_width), -1, dtype=torch.int32, device="cuda"
    )

    seen_widths = []

    def fake_remap(*args, **kwargs):
        seen_widths.append(kwargs["NUM_TOPK_TOKENS"])
        return args[2]

    def fake_forward(self, q, kv, topk_indices, attn_metadata):
        assert topk_indices.shape == (num_tokens, buffer_width)
        return q

    monkeypatch.setattr(
        base_module, "triton_convert_req_index_to_global_index", fake_remap
    )
    impl._forward_bf16_kv = MethodType(fake_forward, impl)

    q = torch.zeros(num_tokens, impl.num_heads, _DIM_QK, device="cuda")
    kv = torch.zeros(1, 64, _DIM_QK, device="cuda")
    metadata = SimpleNamespace(
        req_id_per_token=torch.zeros(num_tokens, dtype=torch.int32, device="cuda"),
        block_table=torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
        block_size=64,
        topk_tokens=2048,
    )

    output, lse = impl.forward_mqa(q, kv, metadata, SimpleNamespace())

    assert output is q
    assert lse is None
    assert seen_widths == [buffer_width]
