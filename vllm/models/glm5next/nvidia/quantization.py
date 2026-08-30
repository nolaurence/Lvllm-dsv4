# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
from vllm.model_executor.kernels.linear import init_mxfp4_linear_kernel
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from vllm.model_executor.layers.quantization.online.mxfp4 import (
    Mxfp4OnlineLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import block_dequant
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import mxfp4_quantize
from vllm.model_executor.utils import replace_parameter


class Mxfp4A16OnlineLinearMethod(Fp8LinearMethod):
    """Load block-FP8 checkpoint weights and run them as MXFP4/A16."""

    def __init__(self, checkpoint_quant_config: Fp8Config) -> None:
        super().__init__(checkpoint_quant_config)
        if self.weight_block_size != [128, 128]:
            raise ValueError(
                "GLM-5 attention W4A16 requires block-FP8 weights with "
                "weight_block_size=[128, 128]"
            )
        self.kernel = init_mxfp4_linear_kernel()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        assert self.weight_block_size is not None
        weight = block_dequant(
            layer.weight,
            layer.weight_scale_inv,
            self.weight_block_size,
        ).to(self.input_dtype)
        weight_fp4, weight_scale = mxfp4_quantize(weight.contiguous())

        layer.input_scale = None
        replace_parameter(layer, "weight", weight_fp4.data)
        replace_parameter(layer, "weight_scale", weight_scale.data)
        del layer._parameters["weight_scale_inv"]
        self.kernel.process_weights_after_loading(layer)
        layer._already_called_process_weights_after_loading = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


class Glm5NextAttentionW4A16Config(QuantizationConfig):
    def __init__(
        self,
        checkpoint_quant_config: Fp8Config,
        checkpoint_weights_are_fp8: bool,
    ) -> None:
        super().__init__()
        self.checkpoint_quant_config = checkpoint_quant_config
        self.checkpoint_weights_are_fp8 = checkpoint_weights_are_fp8

    @classmethod
    def get_name(cls) -> str:
        return "mxfp4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, LinearBase):
            if self.checkpoint_weights_are_fp8:
                return Mxfp4A16OnlineLinearMethod(self.checkpoint_quant_config)
            return Mxfp4OnlineLinearMethod()
        return None


def get_glm5_next_attention_quant_config(
    checkpoint_quant_config: QuantizationConfig | None,
    *,
    checkpoint_weights_are_fp8: bool = True,
) -> QuantizationConfig | None:
    if not envs.LVLLM_GLM5_ATTN_W4A16:
        return None
    if not isinstance(checkpoint_quant_config, Fp8Config):
        raise RuntimeError(
            "GLM-5 attention W4A16 requires an FP8 checkpoint quantization config"
        )
    if checkpoint_quant_config.weight_block_size != [128, 128]:
        raise RuntimeError(
            "GLM-5 attention W4A16 requires weight_block_size=[128, 128]"
        )
    return Glm5NextAttentionW4A16Config(
        checkpoint_quant_config,
        checkpoint_weights_are_fp8,
    )
