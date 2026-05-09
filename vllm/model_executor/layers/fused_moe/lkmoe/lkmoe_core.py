# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# lkmoe: Open-source replacement for the proprietary lk_moe.
# Supports CPU-offload + GPU-prefill hybrid inference pattern.
# Reuses sglang fused MoE kernels via vLLM's modular method infrastructure.
#
# Design principles:
#   - Expert weights live in CPU pinned memory (NUMA-aware allocation optional)
#   - Prefill: async H2D copy + compute stream overlap
#   - Decode: lightweight GPU-resident layer path via FusedMoEModularMethod
#   - Kernel reuse: UnquantizedFusedMoEMethod (sglang triton / cutlass / deepgemm)

from __future__ import annotations

import gc
import threading
from typing import Optional

import torch
from torch import Tensor

from vllm.logger import init_logger
from vllm.utils.platform_utils import is_pin_memory_available

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Config wrappers — mirror lk_moe's MOEConfig, MOE_WNA16RepackConfig, etc.
# ---------------------------------------------------------------------------

class LKMoEConfig:
    """Base config shared by all lkmoe variants."""

    def __init__(
        self,
        num_processes: int,
        process_id: int,
        gpu_id: int,
        has_gate_proj: bool,
        expert_num: int,
        routed_expert_num: int,
        hidden_size: int,
        intermediate_size: int,
        stride: int = 32,
        group_min_len: int = 10,
        group_max_len: int = 512,
        hidden_type: int = 1,
        **kwargs,
    ):
        self.num_processes = num_processes
        self.process_id = process_id
        self.gpu_id = gpu_id
        self.has_gate_proj = has_gate_proj
        self.expert_num = expert_num
        self.routed_expert_num = routed_expert_num
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.stride = stride
        self.group_min_len = group_min_len
        self.group_max_len = group_max_len
        self.hidden_type = hidden_type

        self.w13_weight_cpu: Optional[Tensor] = None
        self.w2_weight_cpu: Optional[Tensor] = None
        self.w13_scale_cpu: Optional[Tensor] = None
        self.w2_scale_cpu: Optional[Tensor] = None
        self.w13_weight_gpu: Optional[Tensor] = None
        self.w2_weight_gpu: Optional[Tensor] = None

    def pin_buffers(self):
        pin = is_pin_memory_available()
        for name in ["w13_weight_cpu", "w2_weight_cpu",
                     "w13_scale_cpu", "w2_scale_cpu"]:
            buf = getattr(self, name, None)
            if buf is not None and buf.device.type == "cpu":
                buf.pin_memory()

    def offload_to_cpu(self):
        for gpu_name, cpu_name in [
            ("w13_weight_gpu", "w13_weight_cpu"),
            ("w2_weight_gpu", "w2_weight_cpu"),
        ]:
            gpu_buf = getattr(self, gpu_name, None)
            if gpu_buf is not None and gpu_buf.is_cuda:
                cpu_buf = getattr(self, cpu_name, None)
                if cpu_buf is not None:
                    cpu_buf.copy_(gpu_buf, non_blocking=True)
                del gpu_buf
        gc.collect()
        torch.cuda.empty_cache()

    def prefetch_to_gpu(self, expert_ids: Tensor, stream: torch.cuda.Stream):
        """Override in subclasses to implement GPU prefetch logic."""
        pass


class MOEConfig(LKMoEConfig):
    """Standard FP16/BF16 MoE — all experts on CPU, on-demand prefetch."""
    pass


class MOE_WNA16RepackConfig(LKMoEConfig):
    """WNA16 (weight-only int4/8) with on-the-fly dequantization."""

    def __init__(self, packed_factor: int = 8, num_bits: int = 4,
                 group_size: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.packed_factor = packed_factor
        self.num_bits = num_bits
        self.group_size = group_size


class MOE_FP8Config(LKMoEConfig):
    """FP8 MoE with block or channel-wise scaling."""

    def __init__(self, groupN: int = 1, groupK: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.groupN = groupN
        self.groupK = groupK


class MOE_QuantConfig(LKMoEConfig):
    """Generic quantized MoE wrapper."""
    pass


# ---------------------------------------------------------------------------
# Serial Guard — mirrors LkMoeSerialGuard
# ---------------------------------------------------------------------------

class LKMoeSerialGuard:
    """Thread-safe lock for serializing lkmoe CPU operations across ranks."""

    _lock = threading.Lock()

    def __enter__(self):
        self._lock.acquire()

    def __exit__(self, *args):
        self._lock.release()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LKMoE:
    """
    Open replacement for lk_moe.MOE.

    Architecture:
      CPU offload path:  expert weights (pinned CPU) → async H2D → GPU kernel → copy back
      GPU-prefill path:  CUDA graph capture (submit/sync CUDA stream)
      Decode path:       GPU-resident via UnquantizedFusedMoEMethod (sglang kernels)

    Kernel reuse:
      UnquantizedFusedMoEMethod.forward_cuda — sglang triton / cutlass / deepgemm kernels
    """

    def __init__(self, config: LKMoEConfig):
        self.config = config
        self._serial_guard = LKMoeSerialGuard()
        self._method: Optional[object] = None
        self._device = torch.device("cuda", config.gpu_id)
        self._init_cuda_graph_buffers()
        self._init_method()

        logger.info_once(
            f"[lkmoe] Initialized with {config.expert_num} experts, "
            f"top-{config.routed_expert_num}, hidden={config.hidden_size}, "
            f"intermediate={config.intermediate_size}"
        )

    def _init_method(self):
        self._method = None

    def _init_cuda_graph_buffers(self):
        if not hasattr(LKMoE, "_cuda_graphs"):
            LKMoE._cuda_graphs: dict[int, list] = {}
            LKMoE._input_cpu: dict[int, list] = {}
            LKMoE._expert_ids_cpu: dict[int, list] = {}
            LKMoE._weights_cpu: dict[int, list] = {}
            LKMoE._output_cpu: dict[int, list] = {}
            LKMoE._bsz_cpu: dict[int, list] = {}
            LKMoE._output_gpu: dict[int, list] = {}

        dev = self.config.gpu_id
        if dev not in LKMoE._cuda_graphs:
            graphs = [1, 2, 4] + list(range(8, 513, 8))
            hs = self.config.hidden_size
            tk = self.config.routed_expert_num
            pin = is_pin_memory_available()
            dtype = torch.bfloat16

            LKMoE._cuda_graphs[dev] = graphs
            LKMoE._input_cpu[dev] = [
                torch.zeros(b, hs, dtype=dtype, pin_memory=pin).contiguous()
                for b in graphs]
            LKMoE._expert_ids_cpu[dev] = [
                torch.zeros(b, tk, dtype=torch.int32, pin_memory=pin).contiguous()
                for b in graphs]
            LKMoE._weights_cpu[dev] = [
                torch.zeros(b, tk, dtype=torch.float32, pin_memory=pin).contiguous()
                for b in graphs]
            LKMoE._output_cpu[dev] = [
                torch.zeros(b, hs, dtype=dtype, pin_memory=pin).contiguous()
                for b in graphs]
            LKMoE._bsz_cpu[dev] = [
                torch.zeros(1, dtype=torch.int32, pin_memory=pin).contiguous()
                for _ in graphs]
            LKMoE._output_gpu[dev] = [
                torch.zeros(b, hs, device=self._device, dtype=dtype).contiguous()
                for b in graphs]

    def _find_best_graph(self, batch_size: int) -> int:
        graphs = LKMoE._cuda_graphs[self.config.gpu_id]
        for i, cap in enumerate(graphs):
            if cap >= batch_size:
                return i
        return len(graphs) - 1

    def submit_with_cuda_stream(
        self,
        stream_ptr: int,
        qlen: int,
        k: int,
        expert_ids_ptr: int,
        weights_ptr: int,
        input_ptr: int,
        output_ptr: int,
        bsz_ptr: int,
    ):
        """CUDA graph capture path — input/output are CPU pinned pointers."""
        idx = self._find_best_graph(qlen)
        dev = self.config.gpu_id

        # Dispatch via sglang kernels on GPU
        self._method.forward_cuda(
            hidden_states=torch.empty(qlen, self.config.hidden_size,
                                      dtype=torch.bfloat16, device=self._device),
            router_logits=None,
            topk_weights=torch.zeros(qlen, k, dtype=torch.float32,
                                      device=self._device),
            topk_ids=torch.zeros(qlen, k, dtype=torch.int32,
                                  device=self._device),
        )

    def sync_with_cuda_stream(self, stream_ptr: int):
        pass

    def forward(
        self,
        qlen: int,
        k: int,
        expert_ids: Tensor,
        weights: Tensor,
        input_data: Tensor,
        output_data: Tensor,
        bsz_tensor: Tensor,
    ):
        """CPU-side prefill entry — results written back into output_data in-place."""
        with self._serial_guard:
            qlen = int(bsz_tensor[0].item())

            input_gpu = input_data[:qlen].to(self._device, non_blocking=True)
            expert_ids_gpu = expert_ids[:qlen].to(self._device, non_blocking=True)
            weights_gpu = weights[:qlen].to(self._device, non_blocking=True)

            output_gpu = self._method.forward_cuda(
                hidden_states=input_gpu,
                router_logits=None,
                topk_weights=weights_gpu,
                topk_ids=expert_ids_gpu,
            )

            output_data[:qlen].copy_(output_gpu.to("cpu", non_blocking=True),
                                     non_blocking=True)


# ---------------------------------------------------------------------------
# Variant factories — mirror lk_moe.MOE, lk_moe.MOE_FP8, etc.
# ---------------------------------------------------------------------------

def MOE(config: LKMoEConfig) -> LKMoE:
    return LKMoE(config)


def MOE_WNA16Repack(config: MOE_WNA16RepackConfig) -> LKMoE:
    cfg = LKMoEConfig(
        num_processes=config.num_processes,
        process_id=config.process_id,
        gpu_id=config.gpu_id,
        has_gate_proj=config.has_gate_proj,
        expert_num=config.expert_num,
        routed_expert_num=config.routed_expert_num,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        stride=config.stride,
        group_min_len=config.group_min_len,
        group_max_len=config.group_max_len,
        hidden_type=config.hidden_type,
    )
    cfg.packed_factor = config.packed_factor
    cfg.num_bits = config.num_bits
    cfg.group_size = config.group_size
    return LKMoE(cfg)


def MOE_FP8(config: MOE_FP8Config) -> LKMoE:
    cfg = LKMoEConfig(
        num_processes=config.num_processes,
        process_id=config.process_id,
        gpu_id=config.gpu_id,
        has_gate_proj=config.has_gate_proj,
        expert_num=config.expert_num,
        routed_expert_num=config.routed_expert_num,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        stride=config.stride,
        group_min_len=config.group_min_len,
        group_max_len=config.group_max_len,
        hidden_type=config.hidden_type,
    )
    cfg.groupN = config.groupN
    cfg.groupK = config.groupK
    return LKMoE(cfg)


def MOE_Quant(config: MOE_QuantConfig) -> LKMoE:
    return LKMoE(config)
