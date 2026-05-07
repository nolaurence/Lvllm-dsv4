# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
lkmoe — open-source replacement for proprietary lk_moe.
Mirrors lk_moe's full public API.

Public API (use in layer.py as replacement for `import lk_moe`):
    from vllm.model_executor.layers.fused_moe.lkmoe import (
        LKMoE, LKMoEConfig, LKMoeSerialGuard,
        MOE, MOE_WNA16Repack, MOE_FP8, MOE_Quant,
        MOEConfig, MOE_WNA16RepackConfig, MOE_FP8Config, MOE_QuantConfig,
    )
"""

from .lkmoe_core import (
    LKMoE,
    LKMoEConfig,
    LKMoeSerialGuard,
    MOE,
    MOE_WNA16Repack,
    MOE_FP8,
    MOE_Quant,
    MOEConfig,
    MOE_WNA16RepackConfig,
    MOE_FP8Config,
    MOE_QuantConfig,
)

__all__ = [
    "LKMoE",
    "LKMoEConfig",
    "LKMoeSerialGuard",
    "MOE",
    "MOE_WNA16Repack",
    "MOE_FP8",
    "MOE_Quant",
    "MOEConfig",
    "MOE_WNA16RepackConfig",
    "MOE_FP8Config",
    "MOE_QuantConfig",
]