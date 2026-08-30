# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ctypes
import gc
import importlib.util
import os
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, ClassVar

import torch

import vllm.envs as envs
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config
from vllm.envs import is_lk_moe_numa_enabled
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)


def _malloc_trim() -> None:
    with suppress(Exception):
        ctypes.CDLL(None).malloc_trim(0)


def _parse_layer_set(spec: str) -> set[int]:
    layers: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_str, end_str = item.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(
                    "Layer range end must be greater than or equal to its start, "
                    f"got {item!r}"
                )
            layers.update(range(start, end + 1))
        else:
            layers.add(int(item))
    return layers


def _layer_id(layer_name: str) -> int | None:
    from vllm.model_executor.models.utils import extract_layer_index

    return extract_layer_index(layer_name)


def should_use_lk_moe(layer_name: str) -> bool:
    if os.getenv("LVLLM_MOE_NUMA_ENABLED", "0") != "1":
        return False
    if not is_lk_moe_numa_enabled():
        return False
    resident_spec = os.getenv("LVLLM_GPU_RESIDENT_MOE_LAYERS", "")
    if not resident_spec:
        return True
    layer_id = _layer_id(layer_name)
    if layer_id is None:
        return True
    try:
        resident_layers = _parse_layer_set(resident_spec)
    except ValueError as err:
        raise ValueError(
            f"Invalid LVLLM_GPU_RESIDENT_MOE_LAYERS={resident_spec!r}"
        ) from err
    if layer_id in resident_layers:
        logger.info(
            "Keeping MoE layer %s resident on GPU; lk::MOE CPU offload is disabled "
            "for this layer.",
            layer_name,
        )
        return False
    return True


def _weight_mode() -> str:
    return os.getenv("LVLLM_MOE_USE_WEIGHT", "INT4").upper()


def _use_int4_weights() -> bool:
    return _weight_mode() == "INT4"


def _load_kt_kernel_ext():
    module = sys.modules.get("kt_kernel_ext")
    if module is not None:
        return module

    override = os.getenv("LVLLM_KT_KERNEL_EXT_PATH")
    if override:
        candidates = [Path(override)]
    else:
        candidates = sorted(
            Path(__file__).resolve().parents[3].glob("kt_kernel_ext*.so")
        )
    if not candidates or not candidates[0].is_file():
        raise RuntimeError(
            "LVLLM_MOE_USE_WEIGHT=FP8 requires a KTransformers main "
            "kt_kernel_ext build. Set LVLLM_KT_KERNEL_EXT_PATH to its .so file."
        )

    spec = importlib.util.spec_from_file_location("kt_kernel_ext", candidates[0])
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load kt_kernel_ext from {candidates[0]}")
    module = importlib.util.module_from_spec(spec)
    old_dlopen_flags = sys.getdlopenflags()
    deepbind = getattr(os, "RTLD_DEEPBIND", 0)
    try:
        sys.setdlopenflags(os.RTLD_NOW | os.RTLD_LOCAL | deepbind)
        spec.loader.exec_module(module)
    finally:
        sys.setdlopenflags(old_dlopen_flags)
    sys.modules["kt_kernel_ext"] = module
    return module


def _moe_stride(hidden_size: int, intermediate_size: int) -> int:
    raw_stride = os.getenv("LVLLM_MOE_STRIDE", "128")
    try:
        stride = int(raw_stride)
    except ValueError:
        stride = 128
    if (
        stride <= 0
        or stride % 32 != 0
        or hidden_size % stride != 0
        or intermediate_size % stride != 0
    ):
        logger.warning(
            "Invalid LVLLM_MOE_STRIDE=%s for hidden_size=%d and "
            "intermediate_size=%d; falling back to 32.",
            raw_stride,
            hidden_size,
            intermediate_size,
        )
        return 32
    return stride


def _group_len(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        value = 0
    if value <= 0:
        logger.warning("Invalid %s=%s; using default %d.", name, raw_value, default)
        return default
    return value


def _glm5_lk_shared_expert_count() -> int:
    if not envs.LVLLM_GLM5_SHARED_EXPERT_CPU:
        return 0
    hf_config = get_current_vllm_config().model_config.hf_config
    if getattr(hf_config, "model_type", None) not in {
        "glm5_next",
        "glm5_next_text",
    }:
        raise RuntimeError(
            "LVLLM_GLM5_SHARED_EXPERT_CPU is only supported by GLM-5-Next"
        )
    count = int(getattr(hf_config, "n_shared_experts", 0) or 0)
    if count != 1:
        raise RuntimeError(
            "GLM shared-expert CPU fusion currently requires n_shared_experts=1, "
            f"got {count}"
        )
    return count


class LkRoutedExperts(RoutedExperts):
    """Routed experts backed by LvLLM's NUMA-aware CPU MoE extension."""

    _lk_deferred_pending: ClassVar[
        dict[tuple[int, int], tuple[int, int, int, bool, Any, torch.Tensor]]
    ] = {}
    _kt_fp8_module: ClassVar[Any | None] = None
    _kt_fp8_cpu_infer: ClassVar[Any | None] = None

    def __init__(self, *args, **kwargs):
        # Quant methods inspect this flag while RoutedExperts creates weights.
        self.use_lk_moe = True
        self.lk_extra_shared_experts = _glm5_lk_shared_expert_count()
        super().__init__(*args, **kwargs)

        self.lk_moe = None
        self.lk_moe_config = None
        self._kt_fp8_enabled = False
        self._lk_cpu_buffers: dict[
            tuple[str, tuple[int, ...], torch.dtype], torch.Tensor
        ] = {}
        self._lk_gpu_buffers: dict[
            tuple[str, tuple[int, ...], torch.dtype, int], torch.Tensor
        ] = {}
        self._lk_decode_bridge_logged = False
        self._lk_deferred_logged = False

        vllm_config = get_current_vllm_config()
        hf_config = getattr(vllm_config.model_config, "hf_config", None)
        num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)
        raw_deferred = os.getenv("LVLLM_LK_DEFERRED_EXPERTS")
        try:
            deferred_experts = int(raw_deferred) if raw_deferred else 0
        except ValueError:
            deferred_experts = 0
        self._lk_max_deferred_experts = max(0, min(deferred_experts, self.top_k))
        if self.lk_extra_shared_experts and self._lk_max_deferred_experts:
            raise RuntimeError(
                "GLM shared-expert CPU fusion is incompatible with "
                "LVLLM_LK_DEFERRED_EXPERTS"
            )
        if num_hidden_layers is not None and self.layer_id == num_hidden_layers - 1:
            self._lk_max_deferred_experts = 0

    @property
    def layer_id(self) -> int | None:
        return _layer_id(self.layer_name)

    @property
    def tp_size(self) -> int:
        return self.moe_config.moe_parallel_config.tp_size

    @property
    def tp_rank(self) -> int:
        return self.moe_config.moe_parallel_config.tp_rank

    @property
    def ep_size(self) -> int:
        return self.moe_config.moe_parallel_config.ep_size

    @property
    def ep_rank(self) -> int:
        return self.moe_config.moe_parallel_config.ep_rank

    def _ensure_moe_quant_config_init(self) -> None:
        if self.lk_moe is None:
            super()._ensure_moe_quant_config_init()

    def process_weights_after_loading(self) -> None:
        if self.lk_moe is None:
            self._maybe_init_lk_moe()

    def _maybe_init_lk_moe(self) -> None:
        weight_mode = _weight_mode()
        if weight_mode not in {"INT4", "FP8"}:
            return
        from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

        if isinstance(self.quant_method, Fp8MoEMethod):
            if not self.quant_method.block_quant:
                raise ValueError(
                    "LK CPU conversion requires block-scaled FP8 MoE weights; "
                    "per-tensor FP8 is not supported."
                )
            try:
                if weight_mode == "FP8":
                    self._process_fp8_block_weights_to_kt_fp8()
                else:
                    self._process_fp8_block_weights_to_lk_int4()
            except Exception as err:
                self.lk_moe = None
                self.lk_moe_config = None
                raise RuntimeError(
                    f"Failed to initialize {weight_mode} CPU MoE"
                ) from err
            return

        if weight_mode == "FP8":
            raise RuntimeError(
                "LVLLM_MOE_USE_WEIGHT=FP8 requires a block-FP8 checkpoint"
            )

        try:
            from vllm.model_executor.layers.quantization.mxfp4 import (
                GptOssMxfp4MoEMethod,
                Mxfp4MoEMethod,
            )

            if isinstance(self.quant_method, (GptOssMxfp4MoEMethod, Mxfp4MoEMethod)):
                self._process_mxfp4_weights_to_lk_int4()
        except Exception as err:
            self.lk_moe = None
            self.lk_moe_config = None
            raise RuntimeError("Failed to initialize MXFP4 lk::MOE") from err

    @classmethod
    def _get_kt_fp8_runtime(cls):
        if cls._kt_fp8_module is None:
            cls._kt_fp8_module = _load_kt_kernel_ext()
        if cls._kt_fp8_cpu_infer is None:
            threads = max(1, int(os.getenv("LK_THREADS", "1")))
            cls._kt_fp8_cpu_infer = cls._kt_fp8_module.CPUInfer(threads)
        return cls._kt_fp8_module, cls._kt_fp8_cpu_infer

    def _process_fp8_block_weights_to_kt_fp8(self) -> None:
        if self.quant_method.weight_block_size != [128, 128]:
            raise ValueError(
                "KTransformers FP8 CPU MoE requires weight_block_size=[128, 128]"
            )

        kt_ext, cpu_infer = self._get_kt_fp8_runtime()
        w13_weight = self.w13_weight
        w2_weight = self.w2_weight
        w13_scale = self.w13_weight_scale_inv
        w2_scale = self.w2_weight_scale_inv

        expert_num, total_intermediate, hidden = w13_weight.shape
        intermediate = total_intermediate // 2
        if total_intermediate % 2 or w2_weight.shape != (
            expert_num,
            hidden,
            intermediate,
        ):
            raise ValueError(
                "Unexpected block-FP8 tensors for KTransformers CPU MoE: "
                f"w13={tuple(w13_weight.shape)}, w2={tuple(w2_weight.shape)}"
            )

        gate_weights = [w13_weight[i, :intermediate] for i in range(expert_num)]
        up_weights = [w13_weight[i, intermediate:] for i in range(expert_num)]
        down_weights = [w2_weight[i] for i in range(expert_num)]
        gate_scales = [
            w13_scale[i, : intermediate // 128].float().contiguous()
            for i in range(expert_num)
        ]
        up_scales = [
            w13_scale[i, intermediate // 128 :].float().contiguous()
            for i in range(expert_num)
        ]
        down_scales = [w2_scale[i].float().contiguous() for i in range(expert_num)]
        if not all(
            tensor.is_contiguous()
            for tensor in (*gate_weights, *up_weights, *down_weights)
        ):
            raise ValueError("KTransformers FP8 expert weights must be contiguous")

        self._kt_fp8_gpu_mask = torch.zeros(expert_num, dtype=torch.bool)
        effective_topk = self.top_k + max(expert_num - self.local_num_experts, 0)
        config = kt_ext.moe.MOEConfig(
            expert_num,
            effective_topk,
            hidden,
            intermediate,
            self._kt_fp8_gpu_mask.data_ptr(),
        )
        config.layer_idx = self.layer_id or 0
        config.pool = cpu_infer.backend_
        config.max_len = max(4, int(envs.LVLLM_KT_FP8_CHUNK_SIZE))
        config.swiglu_limit = float(self.swiglu_limit or 0.0)
        config.quant_config.bits = 8
        config.quant_config.group_size = 128
        config.quant_config.zero_point = False
        config.gate_projs = [[tensor.data_ptr() for tensor in gate_weights]]
        config.up_projs = [[tensor.data_ptr() for tensor in up_weights]]
        config.down_projs = [[tensor.data_ptr() for tensor in down_weights]]
        config.gate_scales = [[tensor.data_ptr() for tensor in gate_scales]]
        config.up_scales = [[tensor.data_ptr() for tensor in up_scales]]
        config.down_scales = [[tensor.data_ptr() for tensor in down_scales]]

        self.lk_moe_config = config
        self._kt_fp8_max_len = config.max_len
        self.lk_moe = kt_ext.moe.AVX2FP8_MOE(config)
        cpu_infer.submit(self.lk_moe.load_weights_task())
        cpu_infer.sync()
        self._kt_fp8_enabled = True

        empty_weight = torch.empty(0, device="cpu", dtype=torch.uint8)
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale_inv",
            "w2_weight_scale_inv",
        ):
            replace_parameter(self, name, empty_weight)
        del w13_weight, w2_weight, w13_scale, w2_scale
        del gate_weights, up_weights, down_weights
        del gate_scales, up_scales, down_scales
        gc.collect()
        _malloc_trim()
        logger.info(
            "Initialized native block-FP8 CPU MoE for layer %s: experts=%d topk=%d",
            self.layer_name,
            expert_num,
            effective_topk,
        )

    @staticmethod
    def _lk_quant_device() -> torch.device:
        if os.getenv("LVLLM_MOE_QUANT_ON_GPU", "0") != "1":
            return torch.device("cpu")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "LVLLM_MOE_QUANT_ON_GPU=1 requires an available CUDA device"
            )
        return torch.device("cuda", torch.cuda.current_device())

    @staticmethod
    def _ggml_type_from_dtype(dtype: torch.dtype) -> int:
        if dtype == torch.float32:
            return 0
        if dtype == torch.float16:
            return 1
        if dtype == torch.bfloat16:
            return 30
        raise ValueError(f"Unsupported dtype {dtype}")

    @staticmethod
    def _dequant_mxfp4_to_bf16(
        weight: torch.Tensor,
        scale: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        weight = weight.to(device, non_blocking=True)
        scale = scale.to(device, non_blocking=True)
        block_size = 32
        n_dim, k_half = weight.shape
        k_dim = k_half * 2
        lut = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.bfloat16,
            device=device,
        )
        scales = torch.pow(2.0, scale.to(torch.float32) - 127.0)
        scales = scales.to(torch.bfloat16).repeat_interleave(block_size, dim=-1)
        if scales.shape[-1] > k_dim:
            scales = scales[..., :k_dim]

        weight_i32 = weight.to(torch.int32)
        out = torch.empty(n_dim, k_dim, dtype=torch.bfloat16, device=device)
        out[:, 0::2] = lut[weight_i32 & 0xF]
        out[:, 1::2] = lut[(weight_i32 >> 4) & 0xF]
        return out.mul_(scales)

    @staticmethod
    def _quantize_rows_q4_0(weight: torch.Tensor) -> torch.Tensor:
        weight = weight.to(torch.float32).contiguous()
        if weight.shape[-1] % 32 != 0:
            raise ValueError(
                "Q4_0 requires the last dimension to be divisible by 32, "
                f"got {weight.shape[-1]}"
            )
        rows = weight.reshape(-1, weight.shape[-1])
        blocks = rows.reshape(rows.shape[0], rows.shape[1] // 32, 32)
        absmax = blocks.abs().amax(dim=-1)
        scale = (absmax / -8.0).to(torch.float16)
        scale_f32 = scale.to(torch.float32)
        safe_scale = torch.where(scale_f32 == 0, torch.ones_like(scale_f32), scale_f32)
        qs = torch.round(blocks / safe_scale.unsqueeze(-1) + 8.0)
        qs = torch.clamp(qs, 0, 15).to(torch.uint8)
        packed = qs[..., :16] | (qs[..., 16:] << 4)
        block_bytes = torch.empty(
            (*blocks.shape[:2], 18), dtype=torch.uint8, device=weight.device
        )
        block_bytes[..., :2] = scale.view(torch.uint8).reshape(*scale.shape, 2)
        block_bytes[..., 2:] = packed
        out_shape = (*weight.shape[:-1], weight.shape[-1] // 32 * 18)
        return block_bytes.reshape(out_shape).contiguous()

    @staticmethod
    def _dequant_fp8_block_weight(
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
        block_shape: list[int],
        device: torch.device,
    ) -> torch.Tensor:
        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError(
                "LK block-FP8 conversion requires float8_e4m3fn weights, "
                f"got {weight.dtype}"
            )
        if len(block_shape) != 2 or any(size <= 0 for size in block_shape):
            raise ValueError(f"Invalid FP8 block shape: {block_shape}")

        block_n, block_k = block_shape
        rows, cols = weight.shape
        expected_scale_shape = (
            (rows + block_n - 1) // block_n,
            (cols + block_k - 1) // block_k,
        )
        if tuple(scale_inv.shape) != expected_scale_shape:
            raise ValueError(
                "Unexpected block-FP8 scale shape: "
                f"weight={tuple(weight.shape)}, scale={tuple(scale_inv.shape)}, "
                f"expected={expected_scale_shape}"
            )

        dequant = weight.to(device, non_blocking=device.type == "cuda").float()
        scale = scale_inv.to(
            device=device, dtype=torch.float32, non_blocking=device.type == "cuda"
        )
        scale = scale.repeat_interleave(block_n, dim=0).repeat_interleave(
            block_k, dim=1
        )
        return dequant.mul_(scale[:rows, :cols])

    def _process_fp8_block_weights_to_lk_int4(self) -> None:
        import vllm._lk_C

        block_shape = self.quant_method.weight_block_size
        if block_shape is None:
            raise ValueError("Block-scaled FP8 weights require weight_block_size")

        w13_weight = self.w13_weight
        w2_weight = self.w2_weight
        w13_scale = self.w13_weight_scale_inv
        w2_scale = self.w2_weight_scale_inv

        expert_num, total_intermediate, hidden = w13_weight.shape
        if total_intermediate % 2 != 0:
            raise ValueError(
                "FP8 w13 output dimension must contain equal gate/up shards, "
                f"got {total_intermediate}"
            )
        intermediate = total_intermediate // 2
        if w2_weight.shape != (expert_num, hidden, intermediate):
            raise ValueError(
                "Unexpected block-FP8 w2 shape for lk::MOE conversion: "
                f"{tuple(w2_weight.shape)}"
            )

        if hidden % 32 != 0 or intermediate % 32 != 0:
            raise ValueError(
                "LK Q4_0 conversion requires hidden and intermediate dimensions "
                f"divisible by 32, got hidden={hidden}, intermediate={intermediate}"
            )
        convert_device = self._lk_quant_device()
        q4_row_bytes_hidden = hidden // 32 * 18
        q4_row_bytes_intermediate = intermediate // 32 * 18

        gate_q4 = torch.empty(
            (expert_num, intermediate, q4_row_bytes_hidden),
            dtype=torch.uint8,
            device="cpu",
        )
        up_q4 = torch.empty_like(gate_q4)
        down_q4 = torch.empty(
            (expert_num, hidden, q4_row_bytes_intermediate),
            dtype=torch.uint8,
            device="cpu",
        )

        for expert_idx in range(expert_num):
            w13_dequant = self._dequant_fp8_block_weight(
                w13_weight[expert_idx],
                w13_scale[expert_idx],
                block_shape,
                convert_device,
            )
            gate_q4[expert_idx].copy_(
                self._quantize_rows_q4_0(w13_dequant[:intermediate]),
                non_blocking=False,
            )
            up_q4[expert_idx].copy_(
                self._quantize_rows_q4_0(w13_dequant[intermediate:]),
                non_blocking=False,
            )
            del w13_dequant

            w2_dequant = self._dequant_fp8_block_weight(
                w2_weight[expert_idx],
                w2_scale[expert_idx],
                block_shape,
                convert_device,
            )
            down_q4[expert_idx].copy_(
                self._quantize_rows_q4_0(w2_dequant), non_blocking=False
            )
            del w2_dequant

        hidden_ggml_type = self._ggml_type_from_dtype(self.moe_config.in_dtype)
        q4_0_type = 2
        group_min_len = _group_len("LVLLM_MOE_GROUP_MIN_LEN", 2)
        group_max_len = _group_len("LVLLM_MOE_GROUP_MAX_LEN", 1024)
        self.lk_moe_config = vllm._lk_C.MOEConfig(
            expert_num,
            self.top_k + max(expert_num - self.local_num_experts, 0),
            self.hidden_size,
            self.intermediate_size_per_partition,
            _moe_stride(self.hidden_size, self.intermediate_size_per_partition),
            group_min_len,
            group_max_len,
            gate_q4.data_ptr(),
            up_q4.data_ptr(),
            down_q4.data_ptr(),
            q4_0_type,
            q4_0_type,
            q4_0_type,
            hidden_ggml_type,
            float(self.swiglu_limit or 0.0),
        )
        self.lk_moe = vllm._lk_C.MOE(self.lk_moe_config)
        del gate_q4, up_q4, down_q4

        empty_weight = torch.empty(0, device="cpu", dtype=torch.uint8)
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale_inv",
            "w2_weight_scale_inv",
        ):
            replace_parameter(self, name, empty_weight)
        del w13_weight, w2_weight, w13_scale, w2_scale
        gc.collect()
        _malloc_trim()
        if convert_device.type == "cuda":
            torch.cuda.empty_cache()
        logger.debug(
            "Initialized block-FP8 lk::MOE Q4_0 weights for layer %s on %s "
            "(group_min_len=%d, group_max_len=%d)",
            self.layer_name,
            convert_device,
            group_min_len,
            group_max_len,
        )

    def _process_mxfp4_weights_to_lk_int4(self) -> None:
        import vllm._lk_C

        w13_weight = self.w13_weight
        w2_weight = self.w2_weight
        w13_scale = self.w13_weight_scale
        w2_scale = self.w2_weight_scale

        expert_num, total_intermediate, hidden_half = w13_weight.shape
        intermediate = total_intermediate // 2
        hidden = hidden_half * 2
        if w2_weight.shape != (expert_num, hidden, intermediate // 2):
            raise ValueError(
                "Unexpected MXFP4 w2 shape for lk::MOE conversion: "
                f"{tuple(w2_weight.shape)}"
            )

        convert_device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        gate_q4 = torch.empty(
            (expert_num, intermediate, hidden // 32 * 18),
            dtype=torch.uint8,
            device="cpu",
        )
        up_q4 = torch.empty_like(gate_q4)
        down_q4 = torch.empty(
            (expert_num, hidden, intermediate // 32 * 18),
            dtype=torch.uint8,
            device="cpu",
        )
        for expert_idx in range(expert_num):
            w13_bf16 = self._dequant_mxfp4_to_bf16(
                w13_weight[expert_idx], w13_scale[expert_idx], convert_device
            )
            gate_q4[expert_idx].copy_(
                self._quantize_rows_q4_0(w13_bf16[:intermediate]),
                non_blocking=False,
            )
            up_q4[expert_idx].copy_(
                self._quantize_rows_q4_0(w13_bf16[intermediate:]),
                non_blocking=False,
            )
            del w13_bf16

            w2_bf16 = self._dequant_mxfp4_to_bf16(
                w2_weight[expert_idx], w2_scale[expert_idx], convert_device
            )
            down_q4[expert_idx].copy_(
                self._quantize_rows_q4_0(w2_bf16), non_blocking=False
            )
            del w2_bf16

        hidden_ggml_type = self._ggml_type_from_dtype(self.moe_config.in_dtype)
        q4_0_type = 2
        group_min_len = _group_len("LVLLM_MOE_GROUP_MIN_LEN", 2)
        group_max_len = _group_len("LVLLM_MOE_GROUP_MAX_LEN", 1024)
        self.lk_moe_config = vllm._lk_C.MOEConfig(
            expert_num,
            self.top_k + max(expert_num - self.local_num_experts, 0),
            self.hidden_size,
            self.intermediate_size_per_partition,
            _moe_stride(self.hidden_size, self.intermediate_size_per_partition),
            group_min_len,
            group_max_len,
            gate_q4.data_ptr(),
            up_q4.data_ptr(),
            down_q4.data_ptr(),
            q4_0_type,
            q4_0_type,
            q4_0_type,
            hidden_ggml_type,
            float(self.swiglu_limit or 0.0),
        )
        self.lk_moe = vllm._lk_C.MOE(self.lk_moe_config)
        del gate_q4, up_q4, down_q4

        empty_weight = torch.empty(0, device="cpu", dtype=torch.uint8)
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
        ):
            replace_parameter(self, name, empty_weight)
        del w13_weight, w2_weight, w13_scale, w2_scale
        gc.collect()
        _malloc_trim()
        if convert_device.type == "cuda":
            torch.cuda.empty_cache()
        logger.debug(
            "Initialized MXFP4 lk::MOE INT4 weights for layer %s "
            "(group_min_len=%d, group_max_len=%d)",
            self.layer_name,
            group_min_len,
            group_max_len,
        )

    def _get_lk_cpu_buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        pin_memory: bool = True,
    ) -> torch.Tensor:
        key = (name, shape, dtype)
        buf = self._lk_cpu_buffers.get(key)
        if buf is None:
            buf = torch.empty(
                shape,
                dtype=dtype,
                device="cpu",
                pin_memory=pin_memory and torch.cuda.is_available(),
            )
            self._lk_cpu_buffers[key] = buf
        return buf

    def _get_lk_gpu_buffer(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        key = (name, shape, dtype, device_index)
        buf = self._lk_gpu_buffers.get(key)
        if buf is None:
            buf = torch.empty(
                shape,
                dtype=dtype,
                device=torch.device("cuda", device_index),
            )
            self._lk_gpu_buffers[key] = buf
        return buf

    def _forward_lk_cuda_decode(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        mapped_topk_ids: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qlen = hidden_states.shape[0]
        k = mapped_topk_ids.shape[1]
        hidden_states = hidden_states.contiguous()
        use_i32_ids = mapped_topk_ids.dtype == torch.int32 and hasattr(
            self.lk_moe, "cpu_decode_i32"
        )
        mapped_topk_ids = mapped_topk_ids.to(
            dtype=torch.int32 if use_i32_ids else torch.int64
        ).contiguous()
        topk_weights = topk_weights.to(dtype=torch.float32).contiguous()

        active_topk_env = os.getenv("LVLLM_LK_DECODE_ACTIVE_TOPK")
        if active_topk_env is not None:
            try:
                active_topk = int(active_topk_env)
            except ValueError:
                active_topk = k
            active_topk = max(0, min(active_topk, k))
            if active_topk < k:
                if not getattr(self, "_lk_decode_active_topk_logged", False):
                    logger.info(
                        "lk::MOE decode for layer %s is masking routed experts "
                        "to active_topk=%d/%d. This is for profiling and changes "
                        "model outputs.",
                        self.layer_name,
                        active_topk,
                        k,
                    )
                    self._lk_decode_active_topk_logged = True
                keep_mask = torch.zeros_like(mapped_topk_ids, dtype=torch.bool)
                if active_topk > 0:
                    keep_pos = torch.topk(
                        topk_weights,
                        k=active_topk,
                        dim=1,
                        largest=True,
                        sorted=False,
                    ).indices
                    keep_mask.scatter_(1, keep_pos, True)
                mapped_topk_ids = mapped_topk_ids.masked_fill(
                    ~keep_mask, -1
                ).contiguous()

        bypass_mode = os.getenv("LVLLM_LK_DECODE_BYPASS")
        if bypass_mode:
            if not getattr(self, "_lk_decode_bypass_logged", False):
                logger.info(
                    "lk::MOE decode for layer %s is bypassed with mode=%s. "
                    "This is for profiling only and changes model outputs.",
                    self.layer_name,
                    bypass_mode,
                )
                self._lk_decode_bypass_logged = True
            if bypass_mode == "identity":
                return hidden_states
            if bypass_mode == "zero":
                return torch.zeros_like(hidden_states)
            if bypass_mode == "empty":
                return torch.empty_like(hidden_states)
            raise RuntimeError(
                "Unsupported LVLLM_LK_DECODE_BYPASS value: "
                f"{bypass_mode!r}; expected identity, zero, or empty"
            )

        stream_ptr = torch.cuda.current_stream(hidden_states.device).cuda_stream
        if output is None:
            output = self._get_lk_gpu_buffer(
                "output",
                tuple(hidden_states.shape),
                hidden_states.dtype,
                hidden_states.device,
            )
        sync_decode_env = os.getenv("LVLLM_LK_CPU_DECODE_SYNC")
        sync_method = "cpu_decode_sync_i32" if use_i32_ids else "cpu_decode_sync"
        use_sync_decode = (
            sync_decode_env == "1" or (sync_decode_env is None and self.tp_size > 1)
        ) and hasattr(self.lk_moe, sync_method)
        if use_sync_decode:
            decode_fn = getattr(self.lk_moe, sync_method)
        else:
            decode_fn = (
                self.lk_moe.cpu_decode_i32 if use_i32_ids else self.lk_moe.cpu_decode
            )
        if (
            not self._lk_decode_bridge_logged
            and os.getenv("LVLLM_LK_LOG_DECODE_BRIDGE", "0") == "1"
        ):
            logger.info(
                "lk::MOE decode bridge for layer %s uses %s "
                "(topk_ids_dtype=%s, expert_map=%s)",
                self.layer_name,
                sync_method
                if use_sync_decode
                else ("cpu_decode_i32" if use_i32_ids else "cpu_decode"),
                mapped_topk_ids.dtype,
                self._expert_map is not None,
            )
            self._lk_decode_bridge_logged = True

        profile_bridge = os.getenv("LVLLM_LK_PROFILE_PY_BRIDGE", "0") == "1"
        if profile_bridge:
            py_bridge_start = time.perf_counter()
        deferred_enabled = (
            self._lk_max_deferred_experts > 0
            and not use_sync_decode
            and hasattr(self.lk_moe, "cpu_decode_nowait")
            and hasattr(self.lk_moe, "cpu_decode_wait")
            and qlen <= int(os.getenv("LVLLM_LK_DEFERRED_MAX_QLEN", "4"))
        )
        device_index = hidden_states.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        pending_key = (device_index, self.tp_rank)
        prev_output = None
        pending = self._lk_deferred_pending.pop(pending_key, None)
        if pending is not None:
            (
                pending_layer,
                pending_qlen,
                pending_k,
                pending_i32,
                pending_moe,
                pending_output,
            ) = pending
            if pending_qlen != qlen:
                raise RuntimeError(
                    "lk::MOE deferred output shape mismatch: pending qlen="
                    f"{pending_qlen}, current qlen={qlen}"
                )
            pending_moe.cpu_decode_wait(
                stream_ptr,
                qlen,
                pending_k,
                pending_i32,
                pending_output.data_ptr(),
            )
            prev_output = pending_output
            if pending_layer + 1 != self.layer_id and not self._lk_deferred_logged:
                logger.warning(
                    "lk::MOE deferred output from layer %s is merged into layer %s. "
                    "This can happen with missing or PP-sharded layers.",
                    pending_layer,
                    self.layer_id,
                )

        if deferred_enabled:
            deferred_count = min(self._lk_max_deferred_experts, max(k - 1, 0))
            if deferred_count > 0:
                protected_k = k - deferred_count
                keep_pos = torch.topk(
                    topk_weights,
                    k=protected_k,
                    dim=1,
                    largest=True,
                    sorted=False,
                ).indices
                keep_mask = torch.zeros_like(mapped_topk_ids, dtype=torch.bool)
                keep_mask.scatter_(1, keep_pos, True)
                immediate_ids = mapped_topk_ids.masked_fill(~keep_mask, -1).contiguous()
                deferred_ids = mapped_topk_ids.masked_fill(keep_mask, -1).contiguous()
                if not self._lk_deferred_logged:
                    logger.info(
                        "lk::MOE deferred decode enabled for layer %s: "
                        "deferred=%d/%d. This changes model outputs.",
                        self.layer_name,
                        deferred_count,
                        k,
                    )
                    self._lk_deferred_logged = True
                decode_fn(
                    stream_ptr,
                    qlen,
                    k,
                    immediate_ids.data_ptr(),
                    topk_weights.data_ptr(),
                    hidden_states.data_ptr(),
                    output.data_ptr(),
                )
                if prev_output is not None:
                    output = output + prev_output
                deferred_output = self._get_lk_gpu_buffer(
                    f"deferred_output_{self.layer_id}",
                    tuple(hidden_states.shape),
                    hidden_states.dtype,
                    hidden_states.device,
                )
                deferred_method = (
                    "cpu_decode_nowait_i32" if use_i32_ids else "cpu_decode_nowait"
                )
                getattr(self.lk_moe, deferred_method)(
                    stream_ptr,
                    qlen,
                    k,
                    deferred_ids.data_ptr(),
                    topk_weights.data_ptr(),
                    hidden_states.data_ptr(),
                    deferred_output.data_ptr(),
                )
                self._lk_deferred_pending[pending_key] = (
                    self.layer_id or 0,
                    qlen,
                    k,
                    use_i32_ids,
                    self.lk_moe,
                    deferred_output,
                )
                return output

        decode_fn(
            stream_ptr,
            qlen,
            k,
            mapped_topk_ids.data_ptr(),
            topk_weights.data_ptr(),
            hidden_states.data_ptr(),
            output.data_ptr(),
        )
        if prev_output is not None:
            output = output + prev_output
        if profile_bridge:
            elapsed_ms = (time.perf_counter() - py_bridge_start) * 1000
            self._lk_py_bridge_calls = getattr(self, "_lk_py_bridge_calls", 0) + 1
            self._lk_py_bridge_ms = getattr(self, "_lk_py_bridge_ms", 0.0) + elapsed_ms
            if self._lk_py_bridge_calls % 32 == 0:
                logger.info(
                    "lk::MOE Python bridge for layer %s calls=%d "
                    "avg_enqueue_ms=%.4f qlen=%d k=%d ids=%s",
                    self.layer_name,
                    self._lk_py_bridge_calls,
                    self._lk_py_bridge_ms / self._lk_py_bridge_calls,
                    qlen,
                    k,
                    "i32" if use_i32_ids else "i64",
                )
        return output

    @eager_break_during_capture
    def _forward_lk_cuda_decode_into(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        mapped_topk_ids: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        effective_num_experts = self.local_num_experts + self.lk_extra_shared_experts
        invalid_mask = (mapped_topk_ids < 0) | (
            mapped_topk_ids >= effective_num_experts
        )
        if torch.any(invalid_mask):
            invalid = mapped_topk_ids[invalid_mask][:16].detach().cpu().tolist()
            raise RuntimeError(
                f"lk::MOE got out-of-range expert ids for layer "
                f"{self.layer_name}: {invalid}; effective_num_experts="
                f"{effective_num_experts}"
            )
        result = self._forward_lk_cuda_decode(
            hidden_states,
            topk_weights,
            mapped_topk_ids,
            output=output,
        )
        if result.data_ptr() != output.data_ptr():
            output.copy_(result)
        return output

    def _append_lk_shared_expert(
        self,
        mapped_topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.lk_extra_shared_experts:
            return mapped_topk_ids, topk_weights
        if self.lk_extra_shared_experts != 1:
            raise RuntimeError("Only one GLM shared expert is currently supported")
        shared_ids = torch.full(
            (mapped_topk_ids.shape[0], 1),
            self.local_num_experts,
            dtype=mapped_topk_ids.dtype,
            device=mapped_topk_ids.device,
        )
        shared_weights = torch.ones(
            (topk_weights.shape[0], 1),
            dtype=topk_weights.dtype,
            device=topk_weights.device,
        )
        return (
            torch.cat((mapped_topk_ids, shared_ids), dim=1),
            torch.cat((topk_weights, shared_weights), dim=1),
        )

    @eager_break_during_capture
    def _forward_kt_fp8_into(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        mapped_topk_ids: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        qlen = hidden_states.shape[0]
        k = mapped_topk_ids.shape[1]
        max_len = self._kt_fp8_max_len
        if max_len <= 0:
            raise RuntimeError("KTransformers FP8 MoE chunk size must be positive")
        effective_num_experts = self.local_num_experts + self.lk_extra_shared_experts
        invalid_mask = (mapped_topk_ids < 0) | (
            mapped_topk_ids >= effective_num_experts
        )
        if torch.any(invalid_mask):
            invalid = mapped_topk_ids[invalid_mask][:16].detach().cpu().tolist()
            raise RuntimeError(
                f"KTransformers FP8 MoE got out-of-range expert ids for layer "
                f"{self.layer_name}: {invalid}; effective_num_experts="
                f"{effective_num_experts}"
            )
        input_cpu = self._get_lk_cpu_buffer(
            "kt_fp8_input", tuple(hidden_states.shape), hidden_states.dtype
        )
        expert_ids_cpu = self._get_lk_cpu_buffer(
            "kt_fp8_expert_ids", tuple(mapped_topk_ids.shape), torch.int64
        )
        weights_cpu = self._get_lk_cpu_buffer(
            "kt_fp8_weights", tuple(topk_weights.shape), torch.float32
        )
        output_cpu = self._get_lk_cpu_buffer(
            "kt_fp8_output", tuple(hidden_states.shape), hidden_states.dtype
        )
        bsz_cpu = self._get_lk_cpu_buffer("kt_fp8_bsz", (1,), torch.int32)
        input_cpu.copy_(hidden_states.detach(), non_blocking=True)
        expert_ids_cpu.copy_(
            mapped_topk_ids.detach().to(torch.int64), non_blocking=True
        )
        weights_cpu.copy_(topk_weights.detach().to(torch.float32), non_blocking=True)
        if hidden_states.is_cuda:
            torch.cuda.current_stream(hidden_states.device).synchronize()

        _, cpu_infer = self._get_kt_fp8_runtime()
        for start in range(0, qlen, max_len):
            end = min(start + max_len, qlen)
            bsz_cpu[0] = end - start
            cpu_infer.submit(
                self.lk_moe.forward_task(
                    bsz_cpu.data_ptr(),
                    k,
                    expert_ids_cpu[start:end].data_ptr(),
                    weights_cpu[start:end].data_ptr(),
                    input_cpu[start:end].data_ptr(),
                    output_cpu[start:end].data_ptr(),
                    False,
                )
            )
            cpu_infer.sync()
        output.copy_(output_cpu, non_blocking=output.is_cuda)
        return output

    def forward_lk(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.lk_moe is None:
            raise RuntimeError("lk::MOE is not initialized")
        qlen = hidden_states.shape[0]

        if self._expert_map is not None:
            topk_ids_i64 = topk_ids.to(torch.int64)
            invalid_mask = (topk_ids_i64 < 0) | (
                topk_ids_i64 >= self._expert_map.numel()
            )
            if torch.any(invalid_mask):
                invalid = topk_ids_i64[invalid_mask][:16].detach().cpu().tolist()
                raise RuntimeError(
                    f"lk::MOE got invalid global expert ids for layer "
                    f"{self.layer_name}: {invalid}; global_num_experts="
                    f"{self._expert_map.numel()}"
                )
            mapped_topk_ids = self._expert_map.to(
                device=topk_ids.device, dtype=torch.int64
            )[topk_ids_i64]
        else:
            mapped_topk_ids = topk_ids

        mapped_topk_ids, topk_weights = self._append_lk_shared_expert(
            mapped_topk_ids, topk_weights
        )

        if getattr(self, "_kt_fp8_enabled", False):
            output = (
                self._get_lk_gpu_buffer(
                    "kt_fp8_output",
                    tuple(hidden_states.shape),
                    hidden_states.dtype,
                    hidden_states.device,
                )
                if hidden_states.is_cuda
                else torch.empty_like(hidden_states)
            )
            return self._forward_kt_fp8_into(
                hidden_states, topk_weights, mapped_topk_ids, output
            )

        if (
            hidden_states.is_cuda
            and os.getenv("LVLLM_DISABLE_LK_CPU_DECODE_BRIDGE", "0") != "1"
            and hasattr(self.lk_moe, "cpu_decode")
        ):
            output = self._get_lk_gpu_buffer(
                "output",
                tuple(hidden_states.shape),
                hidden_states.dtype,
                hidden_states.device,
            )
            return self._forward_lk_cuda_decode_into(
                hidden_states, topk_weights, mapped_topk_ids, output
            )

        k = mapped_topk_ids.shape[1]
        effective_num_experts = self.local_num_experts + self.lk_extra_shared_experts
        invalid_mask = (mapped_topk_ids < 0) | (
            mapped_topk_ids >= effective_num_experts
        )
        if torch.any(invalid_mask):
            invalid = mapped_topk_ids[invalid_mask][:16].detach().cpu().tolist()
            raise RuntimeError(
                f"lk::MOE got out-of-range expert ids for layer "
                f"{self.layer_name}: {invalid}; effective_num_experts="
                f"{effective_num_experts}"
            )

        input_cpu = self._get_lk_cpu_buffer(
            "input", tuple(hidden_states.shape), hidden_states.dtype
        )
        expert_ids_cpu = self._get_lk_cpu_buffer(
            "expert_ids", tuple(mapped_topk_ids.shape), torch.uint64
        )
        weights_cpu = self._get_lk_cpu_buffer(
            "weights", tuple(topk_weights.shape), torch.float32
        )
        output_cpu = self._get_lk_cpu_buffer(
            "output", tuple(hidden_states.shape), hidden_states.dtype
        )
        bsz_cpu = self._get_lk_cpu_buffer("bsz", (1,), torch.int32)
        bsz_cpu[0] = qlen

        input_cpu.copy_(hidden_states.detach(), non_blocking=True)
        expert_ids_cpu.copy_(
            mapped_topk_ids.detach().to(dtype=torch.uint64), non_blocking=True
        )
        weights_cpu.copy_(
            topk_weights.detach().to(dtype=torch.float32), non_blocking=True
        )
        if hidden_states.is_cuda or topk_weights.is_cuda or mapped_topk_ids.is_cuda:
            torch.cuda.current_stream(hidden_states.device).synchronize()

        self.lk_moe.forward(
            qlen,
            k,
            expert_ids_cpu.data_ptr(),
            weights_cpu.data_ptr(),
            input_cpu.data_ptr(),
            output_cpu.data_ptr(),
            bsz_cpu.data_ptr(),
        )
        return output_cpu.to(hidden_states.device, non_blocking=True)
