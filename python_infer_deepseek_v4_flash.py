#!/usr/bin/env python3
"""Project entrypoint for DeepSeek-V4-Flash inference.

This script wraps the reference inference code shipped in
DeepSeek-V4-Flash/inference so the project has a root-level command similar to
python_infer.py. It supports converted checkpoint inference and an optional
conversion step from HuggingFace safetensors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parent
DSV4_DIR = ROOT / "DeepSeek-V4-Flash"
INFER_DIR = DSV4_DIR / "inference"
ENCODING_DIR = DSV4_DIR / "encoding"

sys.path.insert(0, str(INFER_DIR))
sys.path.insert(0, str(ENCODING_DIR))


def convert_checkpoint(args: argparse.Namespace) -> None:
    from convert import main as convert_main

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    convert_main(
        hf_ckpt_path=str(Path(args.model).resolve()),
        save_path=str(save_path.resolve()),
        n_experts=args.n_experts,
        mp=args.model_parallel,
        expert_dtype=args.expert_dtype,
    )


def load_prompts(args: argparse.Namespace) -> List[str]:
    if args.input_file:
        text = Path(args.input_file).read_text(encoding="utf-8")
        return [p for p in text.split("\n\n") if p.strip()]
    return [args.prompt]


def run_inference(args: argparse.Namespace) -> None:
    import torch
    import torch.distributed as dist
    from safetensors.torch import load_model
    from transformers import AutoTokenizer

    from encoding_dsv4 import encode_messages
    from generate import finalize_completion_tokens, generate
    from model import ModelArgs, Transformer

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group("nccl")

    if rank != 0:
        import builtins

        builtins.print = lambda *_, **__: None

    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)

    with open(args.config, encoding="utf-8") as f:
        model_args = ModelArgs(**json.load(f))
    model_args.max_batch_size = args.max_batch_size
    model_args.max_seq_len = args.max_seq_len

    ckpt_path = Path(args.ckpt_path).resolve()
    shard = ckpt_path / f"model{rank}-mp{world_size}.safetensors"
    if not shard.exists():
        raise FileNotFoundError(
            f"converted shard not found: {shard}\n"
            "Run with --convert first, or use:\n"
            "  python DeepSeek-V4-Flash/inference/convert.py --hf-ckpt-path ... --save-path ... --n-experts 256 --model-parallel ..."
        )

    print("========== DeepSeek-V4-Flash config ==========")
    print(f"ckpt_path={ckpt_path}")
    print(f"config={args.config}")
    print(f"world_size={world_size}, rank={rank}, local_rank={local_rank}")
    print(f"max_seq_len={model_args.max_seq_len}, max_batch_size={model_args.max_batch_size}")

    with torch.device("cuda"):
        model = Transformer(model_args)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    print("========== loading model ==========")
    load_model(model, str(shard), strict=False)

    a800_force_dequant = os.getenv("A800_FORCE_DEQUANT_GEMM", "").strip().lower() in {"1", "true", "yes", "on"}
    a800_cuda_fp4 = os.getenv("A800_USE_CUDA_FP4_GEMM", "").strip().lower() in {"1", "true", "yes", "on"}
    a800_cuda_fp4_ffn = os.getenv("A800_USE_CUDA_FP4_FFN", "").strip().lower() in {"1", "true", "yes", "on"}
    a800_cuda_fp4_accum_value = os.getenv("A800_USE_CUDA_FP4_ACCUM", "").strip().lower()
    a800_cuda_fp4_accum = (
        False
        if a800_cuda_fp4_accum_value == ""
        else a800_cuda_fp4_accum_value in {"1", "true", "yes", "on"}
    )
    a800_fast_decode_moe_value = os.getenv("A800_FAST_DECODE_MOE", "").strip().lower()
    a800_fast_decode_moe = (
        a800_force_dequant
        if a800_fast_decode_moe_value == ""
        else a800_fast_decode_moe_value in {"1", "true", "yes", "on"}
    )
    a800_bf16_moe_reduce_value = os.getenv("A800_BF16_MOE_REDUCE", "").strip().lower()
    a800_bf16_moe_reduce = (
        False
        if a800_bf16_moe_reduce_value == ""
        else a800_bf16_moe_reduce_value in {"1", "true", "yes", "on"}
    )
    a800_reuse_decode_moe_y_value = os.getenv("A800_REUSE_DECODE_MOE_Y", "").strip().lower()
    a800_reuse_decode_moe_y = (
        True
        if a800_reuse_decode_moe_y_value == ""
        else a800_reuse_decode_moe_y_value in {"1", "true", "yes", "on"}
    )
    a800_cache_gate_weight_value = os.getenv("A800_CACHE_GATE_WEIGHT_F32", "").strip().lower()
    a800_cache_gate_weight = (
        False
        if a800_cache_gate_weight_value == ""
        else a800_cache_gate_weight_value in {"1", "true", "yes", "on"}
    )
    a800_cache_shared_fp8 = os.getenv("A800_CACHE_SHARED_FP8", "").strip().lower() in {"1", "true", "yes", "on"}
    a800_hash_gate_topk_value = os.getenv("A800_HASH_GATE_TOPK_ONLY", "").strip().lower()
    a800_hash_gate_topk = (
        False
        if a800_hash_gate_topk_value == ""
        else a800_hash_gate_topk_value in {"1", "true", "yes", "on"}
    )
    a800_cache_attn_fp8 = os.getenv("A800_CACHE_ATTN_FP8", "").strip().lower() in {"1", "true", "yes", "on"}
    a800_eos_check_interval = os.getenv("A800_EOS_CHECK_INTERVAL", "").strip()
    a800_defer_token_decode = os.getenv("A800_DEFER_TOKEN_DECODE", "").strip().lower() in {"1", "true", "yes", "on"}
    a800_single_prompt_fast = os.getenv("A800_SINGLE_PROMPT_FAST_GENERATE", "").strip().lower() in {"1", "true", "yes", "on"}
    a800_distributed_argmax = os.getenv("A800_DISTRIBUTED_ARGMAX", "").strip().lower() in {"1", "true", "yes", "on"}
    a800_argmax_gather_tensor = os.getenv("A800_ARGMAX_GATHER_INTO_TENSOR", "").strip().lower() in {"1", "true", "yes", "on"}

    if model_args.scale_dtype == "fp32" or a800_force_dequant:
        import torch.nn as nn

        converted = 0
        for module in model.modules():
            scale = getattr(module, "scale", None)
            weight = getattr(module, "weight", None)
            if isinstance(scale, nn.Parameter) and scale.dtype != torch.float32:
                new_scale = nn.Parameter(scale.detach().float(), requires_grad=False)
                module.scale = new_scale
                if weight is not None and hasattr(weight, "scale"):
                    weight.scale = new_scale
                converted += 1
        print(f"[A800 compat] converted quant scales to fp32: {converted}")

    if a800_force_dequant:
        print(
            "[A800 compat] A800_FORCE_DEQUANT_GEMM=1, "
            "using BF16/FP16 dequantized F.linear fallback instead of TileLang FP8/FP4 GEMM"
        )
    if a800_cuda_fp4:
        print(
            "[A800 compat] A800_USE_CUDA_FP4_GEMM=1, "
            f"trying CUDA .so FP4 expert path: {os.getenv('A800_CUDA_LIB', './build/libdeepseek_v4_a800.so')}"
        )
    if a800_cuda_fp4_ffn:
        print(
            "[A800 compat] A800_USE_CUDA_FP4_FFN=1, "
            "trying CUDA .so two-kernel FP4 expert FFN path "
            f"(fused w1+w3, then w2; direct FP32 MoE accum if available): {os.getenv('A800_CUDA_LIB', './build/libdeepseek_v4_a800.so')}"
        )
    if a800_cuda_fp4_ffn and a800_cuda_fp4_accum:
        print("[A800 compat] A800_USE_CUDA_FP4_ACCUM=1, direct FP32 MoE accumulation is enabled")
    if a800_fast_decode_moe:
        print(
            "[A800 compat] A800_FAST_DECODE_MOE=1, "
            "single-token decode scans selected top-k experts only"
        )
    if a800_bf16_moe_reduce:
        print(
            "[A800 compat] A800_BF16_MOE_REDUCE=1, "
            "MoE routed output accumulates and all-reduces in BF16"
        )
    if a800_fast_decode_moe and not a800_bf16_moe_reduce and a800_reuse_decode_moe_y:
        print(
            "[A800 compat] A800_REUSE_DECODE_MOE_Y=1, "
            "reuse per-layer FP32 MoE decode accumulation buffers"
        )
    if a800_cache_gate_weight:
        print(
            "[A800 compat] A800_CACHE_GATE_WEIGHT_F32=1, "
            "cache per-layer FP32 MoE gate weights after warmup"
        )
    if a800_cache_shared_fp8:
        print(
            "[A800 compat] A800_CACHE_SHARED_FP8=1, "
            "cache BF16 dequantized FP8 shared-expert weights only"
        )
    if a800_hash_gate_topk:
        print(
            "[A800 compat] A800_HASH_GATE_TOPK_ONLY=1, "
            "hash-routed MoE layers score selected experts only"
        )
    if a800_cache_attn_fp8:
        print(
            "[A800 compat] A800_CACHE_ATTN_FP8=1, "
            "cache BF16 dequantized FP8 attention weights only"
        )
    if a800_eos_check_interval != "":
        print(
            f"[A800 compat] A800_EOS_CHECK_INTERVAL={a800_eos_check_interval}, "
            "0 disables per-token EOS synchronization during timing"
        )
    if a800_defer_token_decode:
        print(
            "[A800 compat] A800_DEFER_TOKEN_DECODE=1, "
            "move token tensor CPU conversion and tokenizer decode outside timed region"
        )
    if a800_single_prompt_fast:
        print(
            "[A800 compat] A800_SINGLE_PROMPT_FAST_GENERATE=1, "
            "skip generic batch prompt-mask path for batch=1 throughput tests"
        )
    if a800_distributed_argmax:
        print(
            "[A800 compat] A800_DISTRIBUTED_ARGMAX=1, "
            "greedy decode gathers per-rank max logits instead of full vocab logits"
        )
        if a800_argmax_gather_tensor:
            print(
                "[A800 compat] A800_ARGMAX_GATHER_INTO_TENSOR=1, "
                "reuse a tiny preallocated all_gather_into_tensor buffer for greedy decode"
            )

    torch.set_default_device("cuda")

    prompts = load_prompts(args)
    prompt_tokens = [
        tokenizer.encode(encode_messages([{"role": "user", "content": p}], thinking_mode=args.thinking_mode))
        for p in prompts
    ]

    eos_id = tokenizer.eos_token_id

    print("========== prompt ==========")
    for p in prompts:
        print(p)

    if args.warmup:
        generate(
            model,
            prompt_tokens,
            args.max_new_tokens,
            eos_id,
            args.temperature,
            return_tensor=a800_defer_token_decode,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    bench_iters = max(1, args.bench_iters)
    elapsed_runs = []
    token_result = None
    for _ in range(bench_iters):
        t0 = time.perf_counter()
        token_result = generate(
            model,
            prompt_tokens,
            args.max_new_tokens,
            eos_id,
            args.temperature,
            return_tensor=a800_defer_token_decode,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_runs.append(time.perf_counter() - t0)

    if rank == 0:
        if a800_defer_token_decode:
            tokens, prompt_lens = token_result
            completion_tokens = finalize_completion_tokens(tokens, prompt_lens, args.max_new_tokens, eos_id)
        else:
            completion_tokens = token_result
        completions = tokenizer.batch_decode(completion_tokens)
        new_tokens = sum(len(t) for t in completion_tokens)
        elapsed_best = min(elapsed_runs)
        elapsed_avg = sum(elapsed_runs) / len(elapsed_runs)

        print("========== generated text ==========")
        for i, completion in enumerate(completions):
            if len(completions) > 1:
                print(f"[{i}]")
            print(completion)
            print()

        print("========== performance ==========")
        print(f"generated_tokens={new_tokens}")
        print(f"elapsed_s={elapsed_best:.6f}")
        print(f"tokens_per_s={new_tokens / elapsed_best:.3f}")
        if bench_iters > 1:
            elapsed_text = ",".join(f"{x:.6f}" for x in elapsed_runs)
            print(f"bench_iters={bench_iters}")
            print(f"elapsed_avg_s={elapsed_avg:.6f}")
            print(f"tokens_per_s_avg={new_tokens / elapsed_avg:.3f}")
            print(f"elapsed_runs_s={elapsed_text}")

    if world_size > 1:
        dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DeepSeek-V4-Flash inference wrapper")
    p.add_argument("--model", default="./DeepSeek-V4-Flash", help="HF checkpoint directory before conversion")
    p.add_argument("--ckpt-path", default="./DeepSeek-V4-Flash-converted", help="converted checkpoint directory")
    p.add_argument("--save-path", default="./DeepSeek-V4-Flash-converted", help="output dir for --convert")
    p.add_argument("--config", default="./DeepSeek-V4-Flash/inference/config.json")
    p.add_argument("--prompt", default="黑格尔的哲学思想可以概括为")
    p.add_argument("--input-file", default="")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--max-batch-size", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--thinking-mode", default="chat", choices=["chat", "thinking"])
    p.add_argument("--warmup", action="store_true", help="run one warmup generation before timing")
    p.add_argument("--bench-iters", type=int, default=1, help="repeat timed generation after warmup and report best/avg")
    p.add_argument("--torch-threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=33377335)

    p.add_argument("--convert", action="store_true", help="convert HF safetensors to DeepSeek-V4 inference shards")
    p.add_argument("--n-experts", type=int, default=256)
    p.add_argument("--model-parallel", type=int, default=4)
    p.add_argument("--expert-dtype", choices=["fp4", "fp8"], default=None)
    p.add_argument("--convert-only", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.convert:
        convert_checkpoint(args)
        if args.convert_only:
            return
        args.ckpt_path = args.save_path
    run_inference(args)


if __name__ == "__main__":
    main()
