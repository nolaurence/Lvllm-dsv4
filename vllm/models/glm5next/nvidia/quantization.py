# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import vllm.envs as envs
from vllm.config.quantization import QuantizationConfigArgs, QuantSpec
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8StaticTensorSym,
)


def get_glm5_next_attention_quant_config(
    checkpoint_quant_config: QuantizationConfig | None = None,
    *,
    checkpoint_weights_are_fp8: bool = True,
) -> QuantizationConfig | None:
    del checkpoint_quant_config, checkpoint_weights_are_fp8
    if envs.LVLLM_GLM5_ATTN_W4A16:
        raise RuntimeError(
            "LVLLM_GLM5_ATTN_W4A16 is disabled: GLM-5.3 attention W4A16 "
            "causes deterministic single-token decode collapse"
        )
    if envs.LVLLM_GLM5_ATTN_W8A16:
        # Weight-only FP8 for the attention projections. GLM-5.3-Flash keeps
        # every KDA projection and MLA kv_b_proj in BF16 even in the FP8
        # checkpoint (modules_to_not_convert), and the MLA q_a/kv_a/q_b/o_proj
        # FP8-block weights are dequantized to BF16 on load
        # (see _try_load_fp8_attn_proj). This config re-quantizes all of them
        # per-tensor during loading: Marlin W8A16 on Ampere, FP8 GEMM
        # (W8A8) where the hardware supports it.
        return OnlineQuantizationConfig(
            QuantizationConfigArgs(linear=QuantSpec(weight=kFp8StaticTensorSym))
        )
    return None
