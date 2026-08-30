# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import vllm.envs as envs
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


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
    return None
