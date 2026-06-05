import math
import os
import ctypes
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from functools import lru_cache
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn


world_size = 1
rank = 0
block_size = 128
fp4_block_size = 32
default_dtype = torch.bfloat16
scale_fmt = None
scale_dtype = torch.float32
_FP4_TABLE = None
_A800_FP4_DEQUANT_CACHE = OrderedDict()
_A800_FP4_DEQUANT_CACHE_BYTES = 0
_A800_CUDA_LIB = None
_A800_CUDA_LIB_FAILED = False
_A800_CUDA_FP4_WARNED = False


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _a800_force_dequant_gemm() -> bool:
    return _env_flag("A800_FORCE_DEQUANT_GEMM")


def _a800_cache_dequant_weight() -> bool:
    return _env_flag("A800_DEQUANT_CACHE")


def _a800_cache_shared_fp8_weight() -> bool:
    return _env_flag("A800_CACHE_SHARED_FP8")


def _a800_cache_attn_fp8_weight() -> bool:
    return _env_flag("A800_CACHE_ATTN_FP8")


def _a800_cache_fp4_weight() -> bool:
    return _env_flag("A800_DEQUANT_CACHE_FP4")


def _a800_use_cuda_fp4_gemm() -> bool:
    return _env_flag("A800_USE_CUDA_FP4_GEMM")


def _a800_use_cuda_fp4_ffn() -> bool:
    return _env_flag("A800_USE_CUDA_FP4_FFN")


def _a800_use_cuda_fp4_topk_ffn() -> bool:
    return _env_flag("A800_USE_CUDA_FP4_TOPK_FFN")


def _a800_use_cuda_fp4_accum() -> bool:
    value = os.getenv("A800_USE_CUDA_FP4_ACCUM")
    if value is None or value.strip() == "":
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _a800_fast_decode_moe() -> bool:
    value = os.getenv("A800_FAST_DECODE_MOE")
    if value is None or value.strip() == "":
        return _a800_force_dequant_gemm()
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _a800_bf16_moe_reduce() -> bool:
    value = os.getenv("A800_BF16_MOE_REDUCE")
    if value is None or value.strip() == "":
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _a800_reuse_decode_moe_y() -> bool:
    value = os.getenv("A800_REUSE_DECODE_MOE_Y")
    if value is None or value.strip() == "":
        return True
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _a800_reuse_topk_index_i32() -> bool:
    value = os.getenv("A800_REUSE_TOPK_INDEX_I32")
    if value is None or value.strip() == "":
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _a800_async_moe_allreduce() -> bool:
    value = os.getenv("A800_ASYNC_MOE_ALLREDUCE")
    if value is None or value.strip() == "":
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _a800_cache_gate_weight_f32() -> bool:
    value = os.getenv("A800_CACHE_GATE_WEIGHT_F32")
    if value is None or value.strip() == "":
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _a800_hash_gate_topk_only() -> bool:
    value = os.getenv("A800_HASH_GATE_TOPK_ONLY")
    if value is None or value.strip() == "":
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _a800_argmax_gather_into_tensor() -> bool:
    value = os.getenv("A800_ARGMAX_GATHER_INTO_TENSOR")
    if value is None or value.strip() == "":
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _a800_fp4_cache_limit_bytes() -> int:
    if not _a800_cache_fp4_weight():
        return 0
    value = os.getenv("A800_DEQUANT_CACHE_FP4_MB", "4096").strip()
    try:
        mb = int(value)
    except ValueError:
        mb = 4096
    return max(mb, 0) * 1024 * 1024


def _a800_keep_act_quant() -> bool:
    return _env_flag("A800_KEEP_ACT_QUANT")


def _a800_keep_rotate() -> bool:
    return _env_flag("A800_KEEP_ROTATE")


def _a800_dequant_dtype() -> torch.dtype:
    value = os.getenv("A800_DEQUANT_DTYPE", "bf16").strip().lower()
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32"}:
        return torch.float32
    return torch.bfloat16


def _to_dequant_dtype(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    try:
        return x.to(dtype)
    except RuntimeError:
        return x.float().to(dtype)


def _a800_cuda_lib_path() -> str:
    return os.getenv(
        "A800_CUDA_LIB",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "build", "libdeepseek_v4_a800.so")),
    )


def _a800_load_cuda_lib():
    global _A800_CUDA_LIB, _A800_CUDA_LIB_FAILED

    if _A800_CUDA_LIB is not None:
        return _A800_CUDA_LIB
    if _A800_CUDA_LIB_FAILED:
        return None

    path = _a800_cuda_lib_path()
    try:
        lib = ctypes.CDLL(path)
        lib.ds_v4_fp4_dequant_gemm_bf16.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.ds_v4_fp4_dequant_gemm_bf16.restype = ctypes.c_int
        try:
            lib.ds_v4_fp4_expert_ffn_bf16.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_void_p,
            ]
            lib.ds_v4_fp4_expert_ffn_bf16.restype = ctypes.c_int
        except AttributeError:
            if rank == 0:
                print("[A800 compat] CUDA fp4 expert FFN symbol unavailable in .so")
        try:
            lib.ds_v4_fp4_expert_ffn_accum_f32.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_void_p,
            ]
            lib.ds_v4_fp4_expert_ffn_accum_f32.restype = ctypes.c_int
            if rank == 0:
                print("[A800 compat] CUDA fp4 expert FFN direct-accum symbol available")
        except AttributeError:
            pass
        try:
            lib.ds_v4_fp4_topk_expert_ffn_accum_f32.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_void_p,
            ]
            lib.ds_v4_fp4_topk_expert_ffn_accum_f32.restype = ctypes.c_int
            if rank == 0:
                print("[A800 compat] CUDA fp4 top-k expert FFN symbol available")
        except AttributeError:
            pass
        lib.ds_v4_a800_last_error.argtypes = []
        lib.ds_v4_a800_last_error.restype = ctypes.c_char_p
        _A800_CUDA_LIB = lib
        if rank == 0:
            print(f"[A800 compat] loaded CUDA fp4 gemm lib: {path}")
        return lib
    except OSError as exc:
        _A800_CUDA_LIB_FAILED = True
        if rank == 0:
            print(f"[A800 compat] CUDA fp4 gemm lib unavailable: {path} ({exc})")
        return None


def _a800_warn_cuda_fp4_fallback(message: str) -> None:
    global _A800_CUDA_FP4_WARNED
    if not _A800_CUDA_FP4_WARNED and rank == 0:
        print(f"[A800 compat] CUDA fp4 gemm fallback to PyTorch: {message}")
        _A800_CUDA_FP4_WARNED = True


def _get_fp4_table(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    global _FP4_TABLE
    if _FP4_TABLE is None or _FP4_TABLE.device != device or _FP4_TABLE.dtype != dtype:
        _FP4_TABLE = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            device=device,
            dtype=dtype,
        )
    return _FP4_TABLE


def _tensor_nbytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()


def _a800_fp4_cache_get(weight: torch.Tensor) -> Optional[torch.Tensor]:
    if _a800_fp4_cache_limit_bytes() <= 0:
        return None
    key = id(weight)
    item = _A800_FP4_DEQUANT_CACHE.get(key)
    if item is None:
        return None
    _A800_FP4_DEQUANT_CACHE.move_to_end(key)
    return item[0]


def _a800_fp4_cache_put(weight: torch.Tensor, dequant: torch.Tensor) -> None:
    global _A800_FP4_DEQUANT_CACHE_BYTES

    limit = _a800_fp4_cache_limit_bytes()
    if limit <= 0:
        return

    nbytes = _tensor_nbytes(dequant)
    if nbytes > limit:
        return

    key = id(weight)
    old = _A800_FP4_DEQUANT_CACHE.pop(key, None)
    if old is not None:
        _A800_FP4_DEQUANT_CACHE_BYTES -= old[1]

    _A800_FP4_DEQUANT_CACHE[key] = (dequant, nbytes)
    _A800_FP4_DEQUANT_CACHE_BYTES += nbytes

    while _A800_FP4_DEQUANT_CACHE_BYTES > limit and _A800_FP4_DEQUANT_CACHE:
        _, (_, evicted_bytes) = _A800_FP4_DEQUANT_CACHE.popitem(last=False)
        _A800_FP4_DEQUANT_CACHE_BYTES -= evicted_bytes


def _apply_k_block_scales(
    weight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    out_features, in_features = weight.shape
    n_k_blocks = (in_features + group_size - 1) // group_size

    if in_features % group_size == 0 and scales.ndim == 2 and scales.size(1) >= n_k_blocks:
        row_scales = scales[:, :n_k_blocks]
        if row_scales.size(0) != out_features:
            row_scales = row_scales.repeat_interleave(group_size, dim=0)[:out_features]
        if row_scales.size(0) == out_features:
            weight = weight.view(out_features, n_k_blocks, group_size)
            weight.mul_(row_scales[:, :, None])
            return weight.view(out_features, in_features)

    for k_block in range(scales.size(1)):
        start = k_block * group_size
        end = min(start + group_size, in_features)
        if start >= end:
            break
        block_scale = scales[:, k_block]
        if block_scale.numel() != out_features:
            block_scale = block_scale.repeat_interleave(group_size)[:out_features]
        weight[:, start:end].mul_(block_scale[:, None])
    return weight


def _dequantize_fp8_weight(weight: torch.Tensor) -> torch.Tensor:
    should_cache = _a800_cache_dequant_weight() or (
        _a800_cache_shared_fp8_weight()
        and bool(getattr(weight, "_a800_shared_fp8", False))
    ) or (
        _a800_cache_attn_fp8_weight()
        and bool(getattr(weight, "_a800_attn_fp8", False))
    )
    cache = getattr(weight, "_a800_dequant_cache", None)
    if should_cache and cache is not None:
        return cache

    dtype = _a800_dequant_dtype()
    dequant = _to_dequant_dtype(weight, dtype).contiguous()
    scales = _to_dequant_dtype(weight.scale, dtype).contiguous()
    dequant = _apply_k_block_scales(dequant, scales, block_size)

    if should_cache:
        weight._a800_dequant_cache = dequant
    return dequant


def _torch_hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    pre_logits = mixes[..., :hc_mult]
    post_logits = mixes[..., hc_mult:2 * hc_mult]
    comb_logits = mixes[..., 2 * hc_mult:].view(*mixes.shape[:-1], hc_mult, hc_mult)
    comb_base = hc_base[2 * hc_mult:].view(hc_mult, hc_mult)

    pre = torch.sigmoid(pre_logits * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(post_logits * hc_scale[1] + hc_base[hc_mult:2 * hc_mult])
    comb = (comb_logits * hc_scale[2] + comb_base).softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(max(sinkhorn_iters - 1, 0)):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def _torch_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    bsz, seqlen, n_heads, _ = q.shape
    out = torch.empty_like(q)
    qf = q.float()
    kvf = kv.float()
    sink = attn_sink.float()
    for b_idx in range(bsz):
        for s_idx in range(seqlen):
            idx = topk_idxs[b_idx, s_idx]
            idx = idx[idx >= 0]
            if idx.numel() == 0:
                out[b_idx, s_idx].zero_()
                continue
            kv_sel = kvf[b_idx, idx]
            scores = torch.matmul(qf[b_idx, s_idx], kv_sel.transpose(0, 1)) * softmax_scale
            max_score = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink[:, None])
            score_exp = torch.exp(scores - max_score)
            sink_exp = torch.exp(sink[:, None] - max_score)
            denom = score_exp.sum(dim=-1, keepdim=True) + sink_exp
            out[b_idx, s_idx] = torch.matmul(score_exp / denom, kv_sel).to(q.dtype)
    return out


def _dequantize_fp4_weight(weight: torch.Tensor) -> torch.Tensor:
    cache = _a800_fp4_cache_get(weight)
    if cache is not None:
        return cache

    dtype = _a800_dequant_dtype()
    packed = weight.view(torch.uint8)
    table = _get_fp4_table(weight.device, dtype)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    dequant = torch.stack((table[low.long()], table[high.long()]), dim=-1)
    dequant = dequant.reshape(weight.size(0), -1).contiguous()
    scales = _to_dequant_dtype(weight.scale, dtype).contiguous()
    dequant = _apply_k_block_scales(dequant, scales, fp4_block_size)

    _a800_fp4_cache_put(weight, dequant)
    return dequant


def _a800_cuda_fp4_linear(x: torch.Tensor, weight: torch.Tensor) -> Optional[torch.Tensor]:
    if not _a800_use_cuda_fp4_gemm():
        return None
    if _a800_dequant_dtype() != torch.bfloat16:
        _a800_warn_cuda_fp4_fallback("A800_DEQUANT_DTYPE must be bf16")
        return None
    if x.dtype != torch.bfloat16 or not x.is_cuda:
        _a800_warn_cuda_fp4_fallback("input must be CUDA bf16")
        return None
    if weight.dtype != torch.float4_e2m1fn_x2 or not weight.is_cuda:
        _a800_warn_cuda_fp4_fallback("weight must be CUDA fp4")
        return None

    scale = getattr(weight, "scale", None)
    if scale is None or scale.dtype != torch.float32 or not scale.is_cuda:
        _a800_warn_cuda_fp4_fallback("weight.scale must be CUDA fp32")
        return None
    if scale.ndim != 2:
        _a800_warn_cuda_fp4_fallback("weight.scale must be 2D")
        return None

    lib = _a800_load_cuda_lib()
    if lib is None:
        return None

    in_dim = x.size(-1)
    out_dim = weight.size(0)
    packed_cols = weight.size(1)
    if packed_cols * 2 != in_dim:
        _a800_warn_cuda_fp4_fallback("packed weight shape does not match input dim")
        return None

    x_2d = x.reshape(-1, in_dim).contiguous()
    weight_c = weight.contiguous()
    scale_c = scale.contiguous()
    y_2d = torch.empty((x_2d.size(0), out_dim), device=x.device, dtype=x.dtype)
    stream = torch.cuda.current_stream(x.device).cuda_stream

    ret = lib.ds_v4_fp4_dequant_gemm_bf16(
        ctypes.c_void_p(x_2d.data_ptr()),
        ctypes.c_void_p(weight_c.data_ptr()),
        ctypes.c_void_p(scale_c.data_ptr()),
        ctypes.c_void_p(y_2d.data_ptr()),
        ctypes.c_int(x_2d.size(0)),
        ctypes.c_int(in_dim),
        ctypes.c_int(out_dim),
        ctypes.c_int(scale_c.size(0)),
        ctypes.c_int(scale_c.size(1)),
        ctypes.c_int(fp4_block_size),
        ctypes.c_void_p(stream),
    )
    if ret != 0:
        err = lib.ds_v4_a800_last_error()
        err_text = err.decode("utf-8", errors="replace") if err else f"ret={ret}"
        _a800_warn_cuda_fp4_fallback(err_text)
        return None

    return y_2d.view(*x.shape[:-1], out_dim)


def _a800_cuda_fp4_expert_ffn(
    x: torch.Tensor,
    route_weights: Optional[torch.Tensor],
    w1: "Linear",
    w2: "Linear",
    w3: "Linear",
    swiglu_limit: float,
) -> Optional[torch.Tensor]:
    if not _a800_use_cuda_fp4_ffn():
        return None
    if not _a800_force_dequant_gemm():
        return None
    if route_weights is None:
        return None
    if _a800_dequant_dtype() != torch.bfloat16:
        _a800_warn_cuda_fp4_fallback("A800_DEQUANT_DTYPE must be bf16 for fused FFN")
        return None
    if x.dtype != torch.bfloat16 or not x.is_cuda:
        _a800_warn_cuda_fp4_fallback("fused FFN input must be CUDA bf16")
        return None

    weights = (w1.weight, w2.weight, w3.weight)
    scales = tuple(getattr(weight, "scale", None) for weight in weights)
    if any(weight.dtype != torch.float4_e2m1fn_x2 or not weight.is_cuda for weight in weights):
        _a800_warn_cuda_fp4_fallback("fused FFN requires fp4 CUDA weights")
        return None
    if any(scale is None or scale.dtype != torch.float32 or not scale.is_cuda or scale.ndim != 2 for scale in scales):
        _a800_warn_cuda_fp4_fallback("fused FFN requires CUDA fp32 2D scales")
        return None

    dim = x.size(-1)
    inter_dim = w1.weight.size(0)
    if w3.weight.size(0) != inter_dim or w2.weight.size(0) != dim:
        _a800_warn_cuda_fp4_fallback("fused FFN weight output shapes do not match")
        return None
    if w1.weight.size(1) * 2 != dim or w3.weight.size(1) * 2 != dim or w2.weight.size(1) * 2 != inter_dim:
        _a800_warn_cuda_fp4_fallback("fused FFN packed weight shapes do not match")
        return None

    lib = _a800_load_cuda_lib()
    if lib is None:
        return None
    try:
        ffn = lib.ds_v4_fp4_expert_ffn_bf16
    except AttributeError:
        _a800_warn_cuda_fp4_fallback("fused FFN symbol missing")
        return None

    x_2d = x.reshape(-1, dim).contiguous()
    route = route_weights.reshape(-1)
    if route.dtype != torch.float32:
        route = route.float()
    if not route.is_contiguous():
        route = route.contiguous()
    if route.numel() != x_2d.size(0):
        _a800_warn_cuda_fp4_fallback("route weight count does not match tokens")
        return None

    w1_c, w2_c, w3_c = (weight.contiguous() for weight in weights)
    s1_c, s2_c, s3_c = (scale.contiguous() for scale in scales)
    hidden = torch.empty((x_2d.size(0), inter_dim), device=x.device, dtype=x.dtype)
    y_2d = torch.empty((x_2d.size(0), dim), device=x.device, dtype=x.dtype)
    stream = torch.cuda.current_stream(x.device).cuda_stream

    ret = ffn(
        ctypes.c_void_p(x_2d.data_ptr()),
        ctypes.c_void_p(route.data_ptr()),
        ctypes.c_void_p(w1_c.data_ptr()),
        ctypes.c_void_p(s1_c.data_ptr()),
        ctypes.c_void_p(w2_c.data_ptr()),
        ctypes.c_void_p(s2_c.data_ptr()),
        ctypes.c_void_p(w3_c.data_ptr()),
        ctypes.c_void_p(s3_c.data_ptr()),
        ctypes.c_void_p(0),
        ctypes.c_void_p(hidden.data_ptr()),
        ctypes.c_void_p(y_2d.data_ptr()),
        ctypes.c_int(x_2d.size(0)),
        ctypes.c_int(dim),
        ctypes.c_int(inter_dim),
        ctypes.c_int(s1_c.size(0)),
        ctypes.c_int(s1_c.size(1)),
        ctypes.c_int(s2_c.size(0)),
        ctypes.c_int(s2_c.size(1)),
        ctypes.c_int(s3_c.size(0)),
        ctypes.c_int(s3_c.size(1)),
        ctypes.c_int(fp4_block_size),
        ctypes.c_float(float(swiglu_limit)),
        ctypes.c_void_p(stream),
    )
    if ret != 0:
        err = lib.ds_v4_a800_last_error()
        err_text = err.decode("utf-8", errors="replace") if err else f"ret={ret}"
        _a800_warn_cuda_fp4_fallback(f"fused FFN {err_text}")
        return None

    return y_2d.view(*x.shape[:-1], dim)


def _a800_cuda_fp4_expert_ffn_accum(
    x: torch.Tensor,
    route_weights: Optional[torch.Tensor],
    w1: "Linear",
    w2: "Linear",
    w3: "Linear",
    swiglu_limit: float,
    accum: torch.Tensor,
) -> bool:
    if not _a800_use_cuda_fp4_ffn():
        return False
    if not _a800_use_cuda_fp4_accum():
        return False
    if not _a800_force_dequant_gemm():
        return False
    if route_weights is None:
        return False
    if accum.dtype != torch.float32 or not accum.is_cuda or not accum.is_contiguous():
        return False
    if _a800_dequant_dtype() != torch.bfloat16:
        return False
    if x.dtype != torch.bfloat16 or not x.is_cuda:
        return False

    weights = (w1.weight, w2.weight, w3.weight)
    scales = tuple(getattr(weight, "scale", None) for weight in weights)
    if any(weight.dtype != torch.float4_e2m1fn_x2 or not weight.is_cuda for weight in weights):
        return False
    if any(scale is None or scale.dtype != torch.float32 or not scale.is_cuda or scale.ndim != 2 for scale in scales):
        return False

    dim = x.size(-1)
    inter_dim = w1.weight.size(0)
    if accum.size(-1) != dim:
        return False
    if w3.weight.size(0) != inter_dim or w2.weight.size(0) != dim:
        return False
    if w1.weight.size(1) * 2 != dim or w3.weight.size(1) * 2 != dim or w2.weight.size(1) * 2 != inter_dim:
        return False

    lib = _a800_load_cuda_lib()
    if lib is None:
        return False
    try:
        ffn = lib.ds_v4_fp4_expert_ffn_accum_f32
    except AttributeError:
        return False

    x_2d = x.reshape(-1, dim).contiguous()
    y_2d = accum.reshape(-1, dim)
    if y_2d.size(0) != x_2d.size(0):
        return False

    route = route_weights.reshape(-1)
    if route.dtype != torch.float32:
        route = route.float()
    if not route.is_contiguous():
        route = route.contiguous()
    if route.numel() != x_2d.size(0):
        return False

    w1_c, w2_c, w3_c = (weight.contiguous() for weight in weights)
    s1_c, s2_c, s3_c = (scale.contiguous() for scale in scales)
    hidden = torch.empty((x_2d.size(0), inter_dim), device=x.device, dtype=x.dtype)
    stream = torch.cuda.current_stream(x.device).cuda_stream

    ret = ffn(
        ctypes.c_void_p(x_2d.data_ptr()),
        ctypes.c_void_p(route.data_ptr()),
        ctypes.c_void_p(w1_c.data_ptr()),
        ctypes.c_void_p(s1_c.data_ptr()),
        ctypes.c_void_p(w2_c.data_ptr()),
        ctypes.c_void_p(s2_c.data_ptr()),
        ctypes.c_void_p(w3_c.data_ptr()),
        ctypes.c_void_p(s3_c.data_ptr()),
        ctypes.c_void_p(hidden.data_ptr()),
        ctypes.c_void_p(y_2d.data_ptr()),
        ctypes.c_int(x_2d.size(0)),
        ctypes.c_int(dim),
        ctypes.c_int(inter_dim),
        ctypes.c_int(s1_c.size(0)),
        ctypes.c_int(s1_c.size(1)),
        ctypes.c_int(s2_c.size(0)),
        ctypes.c_int(s2_c.size(1)),
        ctypes.c_int(s3_c.size(0)),
        ctypes.c_int(s3_c.size(1)),
        ctypes.c_int(fp4_block_size),
        ctypes.c_float(float(swiglu_limit)),
        ctypes.c_void_p(stream),
    )
    if ret != 0:
        err = lib.ds_v4_a800_last_error()
        err_text = err.decode("utf-8", errors="replace") if err else f"ret={ret}"
        _a800_warn_cuda_fp4_fallback(f"fused FFN accum {err_text}")
        return False
    return True


def _a800_cuda_fp4_topk_expert_ffn_accum(
    x: torch.Tensor,
    route_weights: torch.Tensor,
    indices: torch.Tensor,
    ptrs: Optional[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Tuple[int, int],
            Tuple[int, int],
            Tuple[int, int],
            int,
        ]
    ],
    local_start: int,
    n_local: int,
    swiglu_limit: float,
    hidden: Optional[torch.Tensor],
    accum: torch.Tensor,
) -> bool:
    if not _a800_use_cuda_fp4_topk_ffn():
        return False
    if not _a800_use_cuda_fp4_ffn():
        return False
    if not _a800_force_dequant_gemm():
        return False
    if ptrs is None:
        return False
    if accum.dtype != torch.float32 or not accum.is_cuda or not accum.is_contiguous():
        return False
    if hidden is None or hidden.dtype != x.dtype or not hidden.is_cuda or not hidden.is_contiguous():
        return False
    if _a800_dequant_dtype() != torch.bfloat16:
        return False
    if x.dtype != torch.bfloat16 or not x.is_cuda:
        return False

    (
        w1_ptrs,
        s1_ptrs,
        w2_ptrs,
        s2_ptrs,
        w3_ptrs,
        s3_ptrs,
        s1_shape,
        s2_shape,
        s3_shape,
        inter_dim,
    ) = ptrs
    ptr_tensors = (w1_ptrs, s1_ptrs, w2_ptrs, s2_ptrs, w3_ptrs, s3_ptrs)
    if any(t.dtype != torch.int64 or not t.is_cuda or not t.is_contiguous() or t.numel() != n_local for t in ptr_tensors):
        return False

    dim = x.size(-1)
    if x.reshape(-1, dim).size(0) != 1:
        return False
    if accum.size(-1) != dim:
        return False
    if inter_dim <= 0 or dim <= 0:
        return False
    if hidden.shape != (route_weights.numel(), inter_dim):
        return False
    if len(s1_shape) != 2 or len(s2_shape) != 2 or len(s3_shape) != 2:
        return False

    route = route_weights.reshape(-1)
    if route.numel() <= 0:
        return False
    if route.dtype != torch.float32:
        route = route.float()
    if not route.is_contiguous():
        route = route.contiguous()

    idx = indices.reshape(-1)
    if idx.numel() != route.numel():
        return False
    if idx.dtype != torch.int32:
        idx = idx.to(torch.int32)
    if not idx.is_contiguous():
        idx = idx.contiguous()

    lib = _a800_load_cuda_lib()
    if lib is None:
        return False
    try:
        ffn = lib.ds_v4_fp4_topk_expert_ffn_accum_f32
    except AttributeError:
        _a800_warn_cuda_fp4_fallback("top-k fused FFN symbol missing")
        return False

    x_2d = x.reshape(-1, dim).contiguous()
    y_2d = accum.reshape(-1, dim)
    if y_2d.size(0) != 1:
        return False

    stream = torch.cuda.current_stream(x.device).cuda_stream

    ret = ffn(
        ctypes.c_void_p(x_2d.data_ptr()),
        ctypes.c_void_p(route.data_ptr()),
        ctypes.c_void_p(idx.data_ptr()),
        ctypes.c_void_p(w1_ptrs.data_ptr()),
        ctypes.c_void_p(s1_ptrs.data_ptr()),
        ctypes.c_void_p(w2_ptrs.data_ptr()),
        ctypes.c_void_p(s2_ptrs.data_ptr()),
        ctypes.c_void_p(w3_ptrs.data_ptr()),
        ctypes.c_void_p(s3_ptrs.data_ptr()),
        ctypes.c_void_p(hidden.data_ptr()),
        ctypes.c_void_p(y_2d.data_ptr()),
        ctypes.c_int(route.numel()),
        ctypes.c_int(local_start),
        ctypes.c_int(n_local),
        ctypes.c_int(dim),
        ctypes.c_int(inter_dim),
        ctypes.c_int(s1_shape[0]),
        ctypes.c_int(s1_shape[1]),
        ctypes.c_int(s2_shape[0]),
        ctypes.c_int(s2_shape[1]),
        ctypes.c_int(s3_shape[0]),
        ctypes.c_int(s3_shape[1]),
        ctypes.c_int(fp4_block_size),
        ctypes.c_float(float(swiglu_limit)),
        ctypes.c_void_p(stream),
    )
    if ret != 0:
        err = lib.ds_v4_a800_last_error()
        err_text = err.decode("utf-8", errors="replace") if err else f"ret={ret}"
        _a800_warn_cuda_fp4_fallback(f"top-k fused FFN accum {err_text}")
        return False
    return True


@contextmanager
def set_dtype(dtype):
    """Temporarily override torch default dtype, restoring it on exit (even if an exception occurs)."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)

@dataclass
class ModelArgs:
    """Model hyperparameters. Field names match the config JSON keys."""
    max_batch_size: int = 4
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "fp8"
    scale_fmt: Literal[None, "ue8m0"] = "ue8m0"
    expert_dtype: Literal[None, "fp4"] = None
    scale_dtype: Literal["fp32", "fp8"] = "fp8"
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 4096
    n_layers: int = 7
    n_hash_layers: int = 0
    n_mtp_layers: int = 1
    n_heads: int = 64
    # moe
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.
    swiglu_limit: float = 0.
    # mqa
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int] = (0, 0, 4, 128, 4, 128, 4, 0)
    # yarn
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    # index
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    # hc
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6


class ParallelEmbedding(nn.Module):
    """Embedding sharded along the vocab dimension. Each rank holds vocab_size // world_size rows.
    Out-of-range indices are zero-masked before all_reduce to combine partial embeddings."""
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Dispatches to fp4_gemm / fp8_gemm / F.linear based on weight dtype.
    For quantized weights, x is first quantized to FP8 via act_quant."""
    assert bias is None

    if weight.dtype == torch.float4_e2m1fn_x2:
        if _a800_force_dequant_gemm():
            y = _a800_cuda_fp4_linear(x, weight)
            if y is not None:
                return y
            return F.linear(x.to(_a800_dequant_dtype()), _dequantize_fp4_weight(weight)).type_as(x)
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp4_gemm(x, s, weight, weight.scale, scale_dtype)
    elif weight.dtype == torch.float8_e4m3fn:
        if _a800_force_dequant_gemm():
            return F.linear(x.to(_a800_dequant_dtype()), _dequantize_fp8_weight(weight)).type_as(x)
        x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
        return fp8_gemm(x, s, weight, weight.scale, scale_dtype)
    else:
        return F.linear(x, weight)


class Linear(nn.Module):
    """Linear layer supporting BF16, FP8, and FP4 weight formats with per-block scaling."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or default_dtype
        if dtype == torch.float4_e2m1fn_x2:
            # FP4: weight is [out, in//2] in float4_e2m1fn_x2, logically [out, in] in fp4
            # Scale is [out, in//32] in float8_e8m0fnu (1 scale per 32 fp4 elements along K)
            self.weight = nn.Parameter(torch.empty(out_features, in_features // 2, dtype=torch.float4_e2m1fn_x2))
            scale_out_features = out_features
            scale_in_features = in_features // fp4_block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float8_e8m0fnu))
        elif dtype == torch.float8_e4m3fn:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float8_e8m0fnu))
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """Shards output dim across TP ranks. No all-reduce needed on output."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class RowParallelLinear(Linear):
    """Shards input dim across TP ranks. All-reduce on output to sum partial results."""
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, None)
        if world_size > 1:
            y = y.float()
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # rmsnorm in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for convenient.
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


@lru_cache(2)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """Precomputes complex exponentials for rotary embeddings with YaRN scaling.
    When original_seq_len > 0, applies frequency interpolation with a smooth
    linear ramp between beta_fast and beta_slow correction ranges."""

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Applies rotary positional embeddings in-place. Uses conjugate for inverse (de-rotation)."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Applies randomized Hadamard rotation to spread information across dims before FP8 quant."""
    assert x.dtype == torch.bfloat16
    if _a800_force_dequant_gemm() and not _a800_keep_rotate():
        return x
    from fast_hadamard_transform import hadamard_transform
    return hadamard_transform(x, scale=x.size(-1) ** -0.5)


@lru_cache(1)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size),  torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(2)
def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


class Compressor(nn.Module):
    """Compresses KV cache via learned gated pooling over `compress_ratio` consecutive tokens.
    When overlap=True (ratio==4), uses overlapping windows for smoother compression boundaries."""

    def __init__(self, args: ModelArgs, compress_ratio: int = 4, head_dim: int = 512, rotate: bool = False):
        super().__init__()
        self.dim = args.dim
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap

        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        # wkv and wgate in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for convenient.
        # When overlap, the first half of dims is for overlapping compression, second half for normal.
        self.wkv = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.wgate = Linear(self.dim, coff * self.head_dim, dtype=torch.float32)
        self.norm = RMSNorm(self.head_dim, args.norm_eps)
        self.kv_cache: torch.Tensor = None  # assigned lazily from Attention.kv_cache
        # State buffers for decode-phase incremental compression.
        # With overlap: state[:, :ratio] = overlapping window, state[:, ratio:] = current window.
        self.register_buffer("kv_state", torch.zeros(args.max_batch_size, coff * compress_ratio, coff * self.head_dim, dtype=torch.float32), persistent=False)
        self.register_buffer("score_state", torch.full((args.max_batch_size, coff * compress_ratio, coff * self.head_dim), float("-inf"), dtype=torch.float32), persistent=False)
        self.freqs_cis: torch.Tensor = None

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        # tensor: [b,s,r,2d]
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        # compression need fp32
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff-ratio : cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff-ratio : cutoff] + self.ape
            if remainder > 0:
                kv, self.kv_state[:bsz, offset : offset+remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:bsz, offset : offset+remainder] = score[:, cutoff:] + self.ape[:remainder]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score += self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return
        kv = self.norm(kv.to(dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        if _a800_force_dequant_gemm() and not _a800_keep_act_quant():
            pass
        elif self.rotate:
            kv = rotate_activation(kv)
            fp4_act_quant(kv, fp4_block_size, True)
        else:
            act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        if start_pos == 0:
            self.kv_cache[:bsz, :seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        return kv


class Indexer(torch.nn.Module):
    """Selects top-k compressed KV positions for sparse attention via learned scoring.
    Has its own Compressor (with Hadamard rotation) to build compressed KV for scoring."""

    def __init__(self, args: ModelArgs, compress_ratio: int = 4):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio

        self.compressor = Compressor(args, compress_ratio, self.head_dim, True)
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len // compress_ratio, self.head_dim), persistent=False)
        self.freqs_cis = None

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)
        # use fp4 simulation for q and kv in indexer
        if not (_a800_force_dequant_gemm() and not _a800_keep_act_quant()):
            fp4_act_quant(q, fp4_block_size, True)
        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        # We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if world_size > 1:
            dist.all_reduce(index_score)
        if start_pos == 0:
            mask = torch.arange(seqlen // ratio).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            index_score += torch.where(mask, float("-inf"), 0)
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs += offset
        return topk_idxs


class Attention(nn.Module):
    """Multi-head Latent Attention (MLA) with sliding window + optional KV compression.
    Uses low-rank Q projection (wq_a -> q_norm -> wq_b) and grouped low-rank O projection."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups // world_size
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))
        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wkv = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(self.n_heads * self.head_dim // self.n_groups, self.n_groups * args.o_lora_rank, dtype=torch.bfloat16)
        self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim)
        self.softmax_scale = self.head_dim ** -0.5
        for linear in (self.wq_a, self.wq_b, self.wkv, self.wo_b):
            linear.weight._a800_attn_fp8 = True

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)
                self.indexer.wq_b.weight._a800_attn_fp8 = True
            else:
                self.indexer = None

        kv_cache_size = args.window_size + (args.max_seq_len // self.compress_ratio if self.compress_ratio else 0)
        self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, kv_cache_size, self.head_dim), persistent=False)
        if self.compress_ratio:
            original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
        else:
            # disable YaRN and use base rope_theta in pure sliding-window attention
            original_seq_len, rope_theta = 0, args.rope_theta
        freqs_cis = precompute_freqs_cis(self.rope_head_dim, args.max_seq_len, original_seq_len,
                                         rope_theta, args.rope_factor, args.beta_fast, args.beta_slow)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis
        # q
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        # win kv & topk_idxs
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        # FP8-simulate non-rope dims to match QAT; rope dims stay bf16 for positional precision
        if not (_a800_force_dequant_gemm() and not _a800_keep_act_quant()):
            act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # compress kv & attn
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                if (kv_compress := self.compressor(x, start_pos)) is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            # We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
            if _a800_force_dequant_gemm():
                o = _torch_sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
            else:
                o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)
            if _a800_force_dequant_gemm():
                o = _torch_sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
            else:
                o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        # o
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        # NOTE: wo_a is FP8 in checkpoint; could do FP8 einsum here for better perf,
        # but using BF16 for simplicity.
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x = self.wo_b(o.flatten(2))
        return x


class Gate(nn.Module):
    """MoE gating: computes expert routing scores and selects top-k experts.
    Supports hash-based routing (first n_hash_layers) where expert indices are
    predetermined per token ID, and score-based routing (remaining layers)."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        if self.hash:
            self.tid2eid = nn.Parameter(torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int32), requires_grad=False)
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))
        self._a800_weight_f32: Optional[torch.Tensor] = None

    def _gate_weight_f32(self) -> torch.Tensor:
        if not _a800_cache_gate_weight_f32() or self.weight.dtype == torch.float32:
            return self.weight.float()
        weight = self._a800_weight_f32
        if weight is None or weight.shape != self.weight.shape or weight.device != self.weight.device:
            weight = self.weight.detach().float().contiguous()
            self._a800_weight_f32 = weight
        return weight

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.hash and self.score_func != "softmax" and _a800_hash_gate_topk_only():
            indices = self.tid2eid[input_ids].long()
            selected_weight = self.weight[indices].float()
            scores = torch.bmm(selected_weight, x.float().unsqueeze(-1)).squeeze(-1)
            if self.score_func == "sigmoid":
                weights = scores.sigmoid()
            else:
                weights = F.softplus(scores).sqrt()
            weights /= weights.sum(dim=-1, keepdim=True)
            weights *= self.route_scale
            return weights, indices

        scores = linear(x.float(), self._gate_weight_f32())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        original_scores = scores
        # Bias shifts scores for expert selection (topk) but does not affect routing weights.
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """Single MoE expert: SwiGLU FFN (w1, w2, w3). Computation in float32 for stability."""
    def __init__(self, dim: int, inter_dim: int, dtype=None, swiglu_limit=0):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, dtype=dtype)
        self.w2 = Linear(inter_dim, dim, dtype=dtype)
        self.w3 = Linear(dim, inter_dim, dtype=dtype)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = _a800_cuda_fp4_expert_ffn(x, weights, self.w1, self.w2, self.w3, self.swiglu_limit)
        if y is not None:
            return y

        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    """Mixture-of-Experts: gate routes each token to top-k routed experts + 1 shared expert.
    Experts are sharded across TP ranks; each rank handles n_routed_experts // world_size experts."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(layer_id, args)
        expert_dtype = torch.float4_e2m1fn_x2 if args.expert_dtype == "fp4" else None
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim, dtype=expert_dtype, swiglu_limit=args.swiglu_limit) if self.experts_start_idx <= i < self.experts_end_idx else None
                                       for i in range(self.n_routed_experts)])
        assert args.n_shared_experts == 1
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)
        for linear in (self.shared_experts.w1, self.shared_experts.w2, self.shared_experts.w3):
            linear.weight._a800_shared_fp8 = True
        self._a800_decode_y: Optional[torch.Tensor] = None
        self._a800_topk_hidden: Optional[torch.Tensor] = None
        self._a800_topk_indices_i32: Optional[torch.Tensor] = None
        self._a800_fp4_topk_ptrs = None
        self._a800_fp4_topk_ptrs_failed = False
        self.swiglu_limit = args.swiglu_limit

    def _a800_decode_accum_buffer(self, x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if dtype != torch.float32 or not _a800_reuse_decode_moe_y():
            return torch.zeros_like(x, dtype=dtype)
        y = self._a800_decode_y
        if y is None or y.shape != x.shape or y.dtype != dtype or y.device != x.device:
            y = torch.empty_like(x, dtype=dtype)
            self._a800_decode_y = y
        y.zero_()
        return y

    def _a800_topk_hidden_buffer(self, x: torch.Tensor, topk: int, inter_dim: int) -> Optional[torch.Tensor]:
        if topk <= 0 or inter_dim <= 0:
            return None
        hidden = self._a800_topk_hidden
        shape = (topk, inter_dim)
        if hidden is None or hidden.shape != shape or hidden.dtype != x.dtype or hidden.device != x.device:
            hidden = torch.empty(shape, device=x.device, dtype=x.dtype)
            self._a800_topk_hidden = hidden
        return hidden

    def _a800_topk_indices_i32_buffer(self, indices: torch.Tensor) -> torch.Tensor:
        idx = indices.reshape(-1)
        if idx.dtype == torch.int32 and idx.is_contiguous():
            return idx
        if not _a800_reuse_topk_index_i32():
            return idx
        buf = self._a800_topk_indices_i32
        if buf is None or buf.numel() != idx.numel() or buf.device != idx.device:
            buf = torch.empty((idx.numel(),), device=idx.device, dtype=torch.int32)
            self._a800_topk_indices_i32 = buf
        buf.copy_(idx, non_blocking=True)
        return buf

    def _a800_local_fp4_topk_ptrs(self):
        if not _a800_use_cuda_fp4_topk_ffn() or self._a800_fp4_topk_ptrs_failed:
            return None
        if self._a800_fp4_topk_ptrs is not None:
            return self._a800_fp4_topk_ptrs

        try:
            w1_ptrs, s1_ptrs = [], []
            w2_ptrs, s2_ptrs = [], []
            w3_ptrs, s3_ptrs = [], []
            s1_shape = s2_shape = s3_shape = None
            inter_dim = None
            device = None

            for expert_id in range(self.experts_start_idx, self.experts_end_idx):
                expert = self.experts[expert_id]
                if expert is None:
                    self._a800_fp4_topk_ptrs_failed = True
                    return None

                weights = (expert.w1.weight, expert.w2.weight, expert.w3.weight)
                scales = tuple(getattr(weight, "scale", None) for weight in weights)
                if any(weight.dtype != torch.float4_e2m1fn_x2 or not weight.is_cuda or not weight.is_contiguous() for weight in weights):
                    self._a800_fp4_topk_ptrs_failed = True
                    return None
                if any(scale is None or scale.dtype != torch.float32 or not scale.is_cuda or not scale.is_contiguous() or scale.ndim != 2 for scale in scales):
                    self._a800_fp4_topk_ptrs_failed = True
                    return None

                dim = self.dim
                cur_inter_dim = expert.w1.weight.size(0)
                if expert.w3.weight.size(0) != cur_inter_dim or expert.w2.weight.size(0) != dim:
                    self._a800_fp4_topk_ptrs_failed = True
                    return None
                if expert.w1.weight.size(1) * 2 != dim or expert.w3.weight.size(1) * 2 != dim or expert.w2.weight.size(1) * 2 != cur_inter_dim:
                    self._a800_fp4_topk_ptrs_failed = True
                    return None

                cur_s1_shape = (scales[0].size(0), scales[0].size(1))
                cur_s2_shape = (scales[1].size(0), scales[1].size(1))
                cur_s3_shape = (scales[2].size(0), scales[2].size(1))
                if s1_shape is None:
                    s1_shape, s2_shape, s3_shape = cur_s1_shape, cur_s2_shape, cur_s3_shape
                    inter_dim = cur_inter_dim
                    device = expert.w1.weight.device
                elif s1_shape != cur_s1_shape or s2_shape != cur_s2_shape or s3_shape != cur_s3_shape or inter_dim != cur_inter_dim:
                    self._a800_fp4_topk_ptrs_failed = True
                    return None

                w1_ptrs.append(expert.w1.weight.data_ptr())
                s1_ptrs.append(scales[0].data_ptr())
                w2_ptrs.append(expert.w2.weight.data_ptr())
                s2_ptrs.append(scales[1].data_ptr())
                w3_ptrs.append(expert.w3.weight.data_ptr())
                s3_ptrs.append(scales[2].data_ptr())

            if device is None or inter_dim is None or s1_shape is None or s2_shape is None or s3_shape is None:
                self._a800_fp4_topk_ptrs_failed = True
                return None

            def ptr_tensor(values):
                return torch.tensor(values, dtype=torch.int64, device=device)

            self._a800_fp4_topk_ptrs = (
                ptr_tensor(w1_ptrs),
                ptr_tensor(s1_ptrs),
                ptr_tensor(w2_ptrs),
                ptr_tensor(s2_ptrs),
                ptr_tensor(w3_ptrs),
                ptr_tensor(s3_ptrs),
                s1_shape,
                s2_shape,
                s3_shape,
                inter_dim,
            )
            return self._a800_fp4_topk_ptrs
        except (RuntimeError, TypeError, ValueError) as exc:
            self._a800_fp4_topk_ptrs_failed = True
            _a800_warn_cuda_fp4_fallback(f"top-k pointer table build failed: {exc}")
            return None

    def _add_shared_expert_after_reduce(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if world_size <= 1:
            y += self.shared_experts(x)
            return y
        if _a800_async_moe_allreduce():
            work = dist.all_reduce(y, async_op=True)
            shared = self.shared_experts(x)
            work.wait()
            y += shared
            return y
        dist.all_reduce(y)
        y += self.shared_experts(x)
        return y

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, input_ids.flatten())
        moe_accum_dtype = x.dtype if _a800_bf16_moe_reduce() else torch.float32

        if _a800_fast_decode_moe() and x.size(0) == 1:
            y = self._a800_decode_accum_buffer(x, moe_accum_dtype)
            topk_ptrs = self._a800_local_fp4_topk_ptrs()
            topk_hidden = self._a800_topk_hidden_buffer(x, weights.numel(), topk_ptrs[-1]) if topk_ptrs is not None else None
            topk_indices = self._a800_topk_indices_i32_buffer(indices) if topk_ptrs is not None else indices
            if _a800_cuda_fp4_topk_expert_ffn_accum(
                x,
                weights,
                topk_indices,
                topk_ptrs,
                self.experts_start_idx,
                self.n_local_experts,
                self.swiglu_limit,
                topk_hidden,
                y,
            ):
                y = self._add_shared_expert_after_reduce(y, x)
                return y.type_as(x).view(shape)
            for top, expert_id in enumerate(indices[0].tolist()):
                if self.experts_start_idx <= expert_id < self.experts_end_idx:
                    expert = self.experts[expert_id]
                    route = weights[:, top : top + 1]
                    if not _a800_cuda_fp4_expert_ffn_accum(
                        x, route, expert.w1, expert.w2, expert.w3, expert.swiglu_limit, y
                    ):
                        y += expert(x, route)
            y = self._add_shared_expert_after_reduce(y, x)
            return y.type_as(x).view(shape)

        y = torch.zeros_like(x, dtype=moe_accum_dtype)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx], weights[idx, top, None])
        y = self._add_shared_expert_after_reduce(y, x)
        return y.type_as(x).view(shape)


class Block(nn.Module):
    """Transformer block with Hyper-Connections (HC) mixing.
    Instead of a simple residual, HC maintains `hc_mult` copies of the hidden state.
    hc_pre: reduces hc copies -> 1 via learned weighted sum (pre-weights from Sinkhorn).
    hc_post: expands 1 -> hc copies via learned post-weights + combination matrix."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc))
            self.hc_attn_scale = nn.Parameter(torch.empty(3))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3))

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        # x: [b,s,hc,d], hc_fn: [mix_hc,hc*d], hc_scale: [3], hc_base: [mix_hc], y: [b,s,hc,d]
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        if _a800_force_dequant_gemm():
            pre, post, comb = _torch_hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
        else:
            pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        # x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc], y: [b,s,hc,d]
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor]) -> torch.Tensor:
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm(x)
        x = self.ffn(x, input_ids)
        x = self.hc_post(x, residual, post, comb)
        return x


class ParallelHead(nn.Module):

    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.part_vocab_size = (vocab_size // world_size)
        # lm_head in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for easier computation of logits later.
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim, dtype=torch.float32))
        self._a800_argmax_local_pack = None
        self._a800_argmax_gather_pack = None

    def get_logits(self, x):
        return F.linear(x[:, -1].float(), self.weight)

    def forward(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, norm: RMSNorm):
        # x: [b,s,hc,d]
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)
        logits = self.get_logits(norm(x))
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits

    def argmax(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, norm: RMSNorm):
        # Greedy decode only needs the global max token. Avoid all-gathering full vocab logits.
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)
        logits = self.get_logits(norm(x))
        local_values, local_indices = logits.max(dim=-1)
        if world_size <= 1:
            return local_indices

        if _a800_argmax_gather_into_tensor() and hasattr(dist, "all_gather_into_tensor"):
            bsz = local_values.numel()
            local_shape = (bsz, 2)
            gather_shape = (world_size * bsz, 2)
            if (
                self._a800_argmax_local_pack is None
                or self._a800_argmax_local_pack.shape != local_shape
                or self._a800_argmax_local_pack.device != local_values.device
                or self._a800_argmax_local_pack.dtype != local_values.dtype
            ):
                self._a800_argmax_local_pack = torch.empty(local_shape, device=local_values.device, dtype=local_values.dtype)
            if (
                self._a800_argmax_gather_pack is None
                or self._a800_argmax_gather_pack.shape != gather_shape
                or self._a800_argmax_gather_pack.device != local_values.device
                or self._a800_argmax_gather_pack.dtype != local_values.dtype
            ):
                self._a800_argmax_gather_pack = torch.empty(gather_shape, device=local_values.device, dtype=local_values.dtype)
            self._a800_argmax_local_pack[:, 0].copy_(local_values)
            self._a800_argmax_local_pack[:, 1].copy_(local_indices.to(local_values.dtype))
            dist.all_gather_into_tensor(self._a800_argmax_gather_pack, self._a800_argmax_local_pack)
            packs = self._a800_argmax_gather_pack.view(world_size, bsz, 2)
        else:
            local_pack = torch.stack((local_values, local_indices.to(local_values.dtype)), dim=-1).contiguous()
            all_packs = [torch.empty_like(local_pack) for _ in range(world_size)]
            dist.all_gather(all_packs, local_pack)
            packs = torch.stack(all_packs, dim=0)
        best_rank = packs[:, :, 0].argmax(dim=0)
        best_local = packs[:, :, 1].gather(0, best_rank.unsqueeze(0)).squeeze(0).to(local_indices.dtype)
        return best_local + best_rank.to(best_local.dtype) * self.part_vocab_size

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)


class MTPBlock(Block):

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        self.e_proj = Linear(args.dim, args.dim)
        self.h_proj = Linear(args.dim, args.dim)
        self.enorm = RMSNorm(args.dim, args.norm_eps)
        self.hnorm = RMSNorm(args.dim, args.norm_eps)
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))
        self.embed: ParallelEmbedding = None
        self.head: ParallelHead = None

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, start_pos: int, input_ids: torch.Tensor) -> torch.Tensor:
        # x: [b,s,hc,d]
        assert self.embed is not None and self.head is not None
        e = self.embed(input_ids)
        e = self.enorm(e)
        x = self.hnorm(x)
        x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)
        x = super().forward(x, start_pos, input_ids)
        logits = self.head(x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
        return logits


class Transformer(nn.Module):
    """Full DeepSeek-V4 model: embed -> HC-expand -> N blocks -> HC-head -> logits.
    Sets global state (world_size, rank, default_dtype, scale_fmt, scale_dtype) in __init__."""
    def __init__(self, args: ModelArgs):
        global world_size, rank, default_dtype, scale_fmt, scale_dtype
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        scale_fmt = "ue8m0" if args.scale_dtype == "fp8" else args.scale_fmt
        scale_dtype = torch.float8_e8m0fnu if args.scale_dtype == "fp8" else torch.float32
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim, self.norm_eps)
        self.head = ParallelHead(args.vocab_size, args.dim, self.norm_eps, self.hc_eps)
        self.mtp = torch.nn.ModuleList()
        for layer_id in range(args.n_mtp_layers):
            self.mtp.append(MTPBlock(args.n_layers + layer_id, args))
            self.mtp[-1].embed = self.embed
            self.mtp[-1].head = self.head
        self.hc_mult = hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
            self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
            self.hc_head_scale = nn.Parameter(torch.empty(1))

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, start_pos: int = 0):
        h = self.embed(input_ids)
        # Expand to hc_mult copies for Hyper-Connections
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.layers:
            h = layer(h, start_pos, input_ids)
        logits = self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
        return logits

    @torch.inference_mode()
    def forward_argmax(self, input_ids: torch.Tensor, start_pos: int = 0):
        h = self.embed(input_ids)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        for layer in self.layers:
            h = layer(h, start_pos, input_ids)
        return self.head.argmax(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs(n_hash_layers=0)
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)

    print(model(x).size())
    for i in range(128, 150):
        print(i, model(x[:, 0:1], i).size())

    h = torch.randn(2, 128, args.hc_mult, args.dim)
    mtp = model.mtp[0]
    print(mtp(h, 0, x).size())
    print(mtp(h[:, 0:1], 1, x[:, 0:1]).size())
