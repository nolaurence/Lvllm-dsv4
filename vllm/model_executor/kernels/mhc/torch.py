# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch


def _finalize_layer_input(
    x_float: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    x = x_float.to(torch.bfloat16)
    if weight is None:
        return x
    variance = x_float.square().mean(dim=-1, keepdim=True)
    weight_float = weight.to(torch.bfloat16).float()
    return (x.float() * torch.rsqrt(variance + eps) * weight_float).to(x.dtype)


def mhc_pre_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor;

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """

    # Validate shapes
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    x = residual_flat.view(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn_flat.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_logits = (
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult :].view(num_tokens, hc_mult, hc_mult) * hc_scale[
        2
    ] + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input_float = torch.sum(
        pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1
    )
    layer_input = _finalize_layer_input(
        layer_input_float,
        norm_weight,
        norm_eps,
    )
    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def mhc_post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh",
        comb_res_mix.to(torch.float32),
        residual.to(torch.float32),
    )
    post_term = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
    return (mixed_residual + post_term).to(residual.dtype)


def mhc_pre_broadcast_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    fn_broadcast: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dim() == 2
    assert norm_weight is not None
    assert fn_broadcast is not None
    hidden_size = residual.shape[-1]
    hc_mult = fn.shape[1] // hidden_size
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = 2 * hc_mult + hc_mult2
    assert fn.shape == (hc_mult3, hc_mult * hidden_size)
    assert fn_broadcast.shape == (hc_mult3, hidden_size)

    residual_float = residual.float()
    mixes = torch.matmul(residual_float, fn_broadcast.t())
    sqrsum = residual_float.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / hidden_size + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps
    post_logits = (
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value
    comb_logits = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult) * hc_scale[
        2
    ] + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    residual_out = residual.unsqueeze(-2).expand(-1, hc_mult, -1).contiguous()
    layer_input_float = pre_mix.sum(dim=-1, keepdim=True) * residual_float
    layer_input = _finalize_layer_input(
        layer_input_float,
        norm_weight,
        norm_eps,
    )
    return (
        residual_out,
        post_mix.unsqueeze(-1),
        comb_mix,
        layer_input,
    )


def mhc_fused_post_pre_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del tile_n
    residual_cur = mhc_post_torch(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
    )
    post_mix, comb_mix, layer_input = mhc_pre_torch(
        residual_cur,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
        norm_weight,
        norm_eps,
    )
    return residual_cur, post_mix, comb_mix, layer_input


def hc_head_fused_torch(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    hc_mult, hidden_size = hidden_states.shape[-2:]
    outer_shape = hidden_states.shape[:-2]
    hs_flat = hidden_states.reshape(-1, hc_mult, hidden_size)
    hs_float = hs_flat.float()
    hs_2d = hs_float.flatten(1)
    rms = torch.rsqrt(hs_2d.square().mean(dim=-1) + rms_norm_eps)
    mixes = torch.matmul(hs_2d, hc_fn.t())
    pre_mix = torch.sigmoid(mixes * rms.unsqueeze(-1) * hc_scale[0] + hc_base) + hc_eps
    out = (pre_mix.unsqueeze(-1) * hs_float).sum(dim=1)
    return out.to(hidden_states.dtype).view(*outer_shape, hidden_size)
