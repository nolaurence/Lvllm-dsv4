# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ctypes
import gc
import os
import time
from contextlib import suppress
from typing import Any, ClassVar

import torch

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


def _use_int4_weights() -> bool:
    return os.getenv("LVLLM_MOE_USE_WEIGHT", "INT4").upper() == "INT4"


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


class LkRoutedExperts(RoutedExperts):
    """Routed experts backed by LvLLM's NUMA-aware CPU MoE extension."""

    _lk_deferred_pending: ClassVar[
        dict[tuple[int, int], tuple[int, int, int, bool, Any, torch.Tensor]]
    ] = {}

    def __init__(self, *args, **kwargs):
        # Quant methods inspect this flag while RoutedExperts creates weights.
        self.use_lk_moe = True
        super().__init__(*args, **kwargs)

        self.lk_moe = None
        self.lk_moe_config = None
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
        if not _use_int4_weights():
            return
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
            self.local_num_experts,
            self.top_k,
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

    def forward_lk(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.lk_moe is None:
            raise RuntimeError("lk::MOE is not initialized")
        qlen = hidden_states.shape[0]
        k = topk_ids.shape[1]

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

        if (
            hidden_states.is_cuda
            and os.getenv("LVLLM_DISABLE_LK_CPU_DECODE_BRIDGE", "0") != "1"
            and hasattr(self.lk_moe, "cpu_decode")
        ):
            return self._forward_lk_cuda_decode(
                hidden_states, topk_weights, mapped_topk_ids
            )

        invalid_mask = (mapped_topk_ids < 0) | (
            mapped_topk_ids >= self.local_num_experts
        )
        if torch.any(invalid_mask):
            invalid = mapped_topk_ids[invalid_mask][:16].detach().cpu().tolist()
            raise RuntimeError(
                f"lk::MOE got out-of-range expert ids for layer "
                f"{self.layer_name}: {invalid}; local_num_experts="
                f"{self.local_num_experts}"
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
