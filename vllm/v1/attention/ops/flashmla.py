# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py

import dataclasses
import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

if current_platform.is_cuda():
    try:
        import vllm._flashmla_C  # noqa: F401

        _flashmla_C_AVAILABLE = True
    except ImportError:
        _flashmla_C_AVAILABLE = False
else:
    _flashmla_C_AVAILABLE = False

if current_platform.is_cuda():
    try:
        import vllm._flashmla_extension_C  # noqa: F401

        _flashmla_extension_C_AVAILABLE = True
    except ImportError:
        _flashmla_extension_C_AVAILABLE = False
else:
    _flashmla_extension_C_AVAILABLE = False


def _is_flashmla_available() -> tuple[bool, str | None]:
    if not _flashmla_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_C is not available, likely was not "
            "compiled due to insufficient nvcc version or a supported arch "
            "was not in the list of target arches to compile for.",
        )
    if not _flashmla_extension_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_extension_C is not available, likely "
            "was not compiled due to a build error.",
        )

    return True, None


def is_flashmla_dense_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    if not current_platform.is_device_capability_family(90):
        return False, "FlashMLA Dense is only supported on Hopper devices."
    return True, None


def is_flashmla_sparse_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    if not (
        current_platform.is_device_capability_family(90)
        or current_platform.is_device_capability_family(100)
    ):
        return (
            False,
            "FlashMLA Sparse is only supported on Hopper and Blackwell DC devices.",
        )
    return True, None


def _raise_flashmla_unavailable(*_args, **_kwargs):
    _, reason = _is_flashmla_available()
    raise RuntimeError(reason or "FlashMLA is not available")


def _use_sm86_flashmla_reference() -> bool:
    forced = os.environ.get("VLLM_SM86_DEEPSEEK_V4_REF", "").strip()
    if forced in ("0", "false", "False"):
        return False
    if forced in ("1", "true", "True"):
        return True
    try:
        return current_platform.is_cuda() and torch.cuda.get_device_capability(0) < (
            9,
            0,
        )
    except Exception:
        return False


if _is_flashmla_available()[0]:
    from vllm.third_party.flashmla.flash_mla_interface import (  # noqa: F401
        FlashMLASchedMeta,
        flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_mla_sparse_fwd,
        flash_mla_with_kvcache,
        get_mla_metadata,
    )
else:

    @dataclasses.dataclass
    class FlashMLASchedMeta:  # type: ignore[no-redef]
        pass

    flash_attn_varlen_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_kvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_attn_varlen_qkvpacked_func = _raise_flashmla_unavailable  # type: ignore[assignment]

    def get_mla_metadata(*_args, **_kwargs):  # type: ignore[no-redef]
        if not _use_sm86_flashmla_reference():
            _raise_flashmla_unavailable()
        return FlashMLASchedMeta(), None

    def _dequant_fp8_kv_slots(
        cache: torch.Tensor,
        slot_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nope_dim = 448
        bf16_dim = 64
        quant_block = 64
        n_nope_blocks = nope_dim // quant_block
        token_data_size = nope_dim + bf16_dim * 2
        token_scale_dim = 8

        num_blocks, block_size = cache.shape[:2]
        max_slot = num_blocks * block_size
        valid_mask = (slot_indices >= 0) & (slot_indices < max_slot)
        safe_idx = torch.where(
            valid_mask, slot_indices, torch.zeros_like(slot_indices)
        ).long()

        block_idx = safe_idx // block_size
        pos_in_block = safe_idx % block_size
        n_slots = safe_idx.shape[0]
        device = cache.device

        cache_flat = cache.reshape(num_blocks, -1)
        block_stride = cache_flat.shape[-1]
        cache_flat_1d = cache_flat.reshape(-1)

        arange_data = torch.arange(token_data_size, device=device)
        arange_scale = torch.arange(n_nope_blocks, device=device)

        data_base = block_idx * block_stride + pos_in_block * token_data_size
        data_idx = (data_base.unsqueeze(-1) + arange_data).flatten()
        data_bytes = cache_flat_1d[data_idx].reshape(n_slots, token_data_size)

        scale_base = (
            block_idx * block_stride
            + block_size * token_data_size
            + pos_in_block * token_scale_dim
        )
        scale_idx = (scale_base.unsqueeze(-1) + arange_scale).flatten()
        scale_bytes = cache_flat_1d[scale_idx].reshape(n_slots, n_nope_blocks)

        fp8_bytes = data_bytes[:, :nope_dim].contiguous()
        bf16_bytes = data_bytes[:, nope_dim : nope_dim + bf16_dim * 2].contiguous()

        fp8_vals_bf = fp8_bytes.view(torch.float8_e4m3fn).to(torch.bfloat16)
        bf16_vals = bf16_bytes.view(torch.bfloat16)
        scales_bf = torch.pow(2.0, scale_bytes.to(torch.float32) - 127.0).to(
            torch.bfloat16
        )
        fp8_grp = fp8_vals_bf.view(
            n_slots, n_nope_blocks, quant_block
        ) * scales_bf.unsqueeze(-1)
        k_bf16 = torch.cat([fp8_grp.reshape(n_slots, nope_dim), bf16_vals], dim=-1)
        k_bf16 = k_bf16 * valid_mask.unsqueeze(-1).to(torch.bfloat16)
        return k_bf16, valid_mask

    def flash_mla_with_kvcache(  # type: ignore[no-redef]
        q: torch.Tensor,
        k_cache: torch.Tensor,
        block_table: torch.Tensor | None,
        cache_seqlens: torch.Tensor | None,
        head_dim_v: int,
        tile_scheduler_metadata: FlashMLASchedMeta,
        num_splits: None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        is_fp8_kvcache: bool = False,
        indices: torch.Tensor | None = None,
        attn_sink: torch.Tensor | None = None,
        extra_k_cache: torch.Tensor | None = None,
        extra_indices_in_kvcache: torch.Tensor | None = None,
        topk_length: torch.Tensor | None = None,
        extra_topk_length: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not _use_sm86_flashmla_reference() or indices is None:
            _raise_flashmla_unavailable()
        assert not causal
        assert is_fp8_kvcache
        assert num_splits is None
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        if out is None:
            out = torch.empty(
                q.shape[0],
                q.shape[1],
                q.shape[2],
                head_dim_v,
                dtype=q.dtype,
                device=q.device,
            )

        bsz, sq, heads, _ = q.shape
        out_view = out.reshape(bsz, sq, heads, head_dim_v)
        lse = torch.zeros(bsz, heads, sq, dtype=torch.float32, device=q.device)
        sink = attn_sink.to(torch.float32) if attn_sink is not None else None

        if indices.dim() == 4:
            indices = indices[:, :, 0, :]
        elif indices.dim() == 3 and indices.shape[1] != sq and indices.shape[1] == 1:
            indices = indices.expand(bsz, sq, indices.shape[-1])

        if extra_indices_in_kvcache is not None:
            if extra_indices_in_kvcache.dim() == 4:
                extra_indices_in_kvcache = extra_indices_in_kvcache[:, :, 0, :]
            elif (
                extra_indices_in_kvcache.dim() == 3
                and extra_indices_in_kvcache.shape[1] == 1
                and sq != 1
            ):
                extra_indices_in_kvcache = extra_indices_in_kvcache.expand(
                    bsz, sq, extra_indices_in_kvcache.shape[-1]
                )

        for b in range(bsz):
            for s in range(sq):
                idx = indices[b, s] if indices.dim() == 3 else indices[b]
                k_main, mask_main = _dequant_fp8_kv_slots(k_cache, idx)
                if topk_length is not None:
                    idx_offsets = torch.arange(idx.shape[0], device=idx.device)
                    mask_main = mask_main & (idx_offsets < topk_length[b])

                if extra_k_cache is not None and extra_indices_in_kvcache is not None:
                    extra_idx = (
                        extra_indices_in_kvcache[b, s]
                        if extra_indices_in_kvcache.dim() == 3
                        else extra_indices_in_kvcache[b]
                    )
                    k_extra, mask_extra = _dequant_fp8_kv_slots(
                        extra_k_cache, extra_idx
                    )
                    if extra_topk_length is not None:
                        extra_offsets = torch.arange(
                            extra_idx.shape[0], device=extra_idx.device
                        )
                        mask_extra = mask_extra & (extra_offsets < extra_topk_length[b])
                    k_all = torch.cat([k_main, k_extra], dim=0)
                    mask = torch.cat([mask_main, mask_extra], dim=0)
                else:
                    k_all = k_main
                    mask = mask_main

                if k_all.shape[0] == 0:
                    out_view[b, s].zero_()
                    continue
                q_bs = q[b, s].to(torch.bfloat16)
                common = min(q_bs.shape[-1], k_all.shape[-1])
                logits = (
                    q_bs[..., :common] @ k_all[..., :common].transpose(-1, -2)
                ).to(torch.float32) * softmax_scale
                logits = logits.masked_fill(~mask.unsqueeze(0), float("-inf"))
                max_logit = logits.amax(dim=-1, keepdim=True)
                max_logit = torch.where(
                    torch.isinf(max_logit), torch.zeros_like(max_logit), max_logit
                )
                exp_logits = (logits - max_logit).exp()
                denom = exp_logits.sum(dim=-1, keepdim=True)
                if sink is not None:
                    denom = denom + (sink.view(heads, 1) - max_logit).exp()
                probs = (exp_logits / denom).to(torch.bfloat16)
                out_view[b, s].copy_((probs @ k_all[:, :head_dim_v]).to(out.dtype))
                lse[b, :, s] = (max_logit.squeeze(-1) + denom.squeeze(-1).log()).to(
                    torch.float32
                )
        return out, lse

    def flash_mla_sparse_fwd(  # type: ignore[no-redef]
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        sm_scale: float,
        d_v: int = 512,
        attn_sink: torch.Tensor | None = None,
        topk_length: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not _use_sm86_flashmla_reference():
            _raise_flashmla_unavailable()
        sq, heads, _ = q.shape
        if out is None:
            out = torch.empty(sq, heads, d_v, dtype=q.dtype, device=q.device)
        max_logits = torch.zeros(sq, heads, dtype=torch.float32, device=q.device)
        lse = torch.zeros(sq, heads, dtype=torch.float32, device=q.device)
        sink = attn_sink.to(torch.float32) if attn_sink is not None else None
        kv_flat = kv[:, 0, :]
        for t in range(sq):
            idx = indices[t, 0]
            valid = (idx >= 0) & (idx < kv_flat.shape[0])
            if topk_length is not None:
                idx_offsets = torch.arange(idx.shape[0], device=idx.device)
                valid = valid & (idx_offsets < topk_length[t])
            safe_idx = torch.where(valid, idx, torch.zeros_like(idx)).long()
            k_all = kv_flat[safe_idx] * valid.unsqueeze(-1).to(kv_flat.dtype)
            common = min(q.shape[-1], k_all.shape[-1])
            logits = (
                q[t, :, :common].to(torch.bfloat16)
                @ k_all[:, :common].transpose(-1, -2)
            ).to(torch.float32) * sm_scale
            logits = logits.masked_fill(~valid.unsqueeze(0), float("-inf"))
            max_logit = logits.amax(dim=-1, keepdim=True)
            max_logit = torch.where(
                torch.isinf(max_logit), torch.zeros_like(max_logit), max_logit
            )
            exp_logits = (logits - max_logit).exp()
            denom = exp_logits.sum(dim=-1, keepdim=True)
            if sink is not None:
                denom = denom + (sink.view(heads, 1) - max_logit).exp()
            probs = (exp_logits / denom).to(torch.bfloat16)
            out[t].copy_((probs @ k_all[:, :d_v]).to(out.dtype))
            max_logits[t] = max_logit.squeeze(-1)
            lse[t] = (max_logit.squeeze(-1) + denom.squeeze(-1).log()).to(torch.float32)
        return out, max_logits, lse


def get_mla_metadata_dense_fp8(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    return torch.ops._flashmla_extension_C.get_mla_decoding_metadata_dense_fp8(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
    )


def flash_mla_with_kvcache_fp8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    descale_q: torch.Tensor | None = None,
    descale_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = torch.ops._flashmla_extension_C.fwd_kvcache_mla_fp8(
        q,
        k_cache,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
        descale_q,
        descale_k,
    )
    return out, softmax_lse
