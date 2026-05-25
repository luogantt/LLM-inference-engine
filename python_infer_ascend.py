#!/usr/bin/env python3
import argparse
import time
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run DeepSeek/Qwen safetensors inference on Ascend NPU.")
    p.add_argument("--model", default="/root/autodl-tmp/deepseek-r1-7b")
    p.add_argument("--prompt", default="你好 deepseek")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-seq", type=int, default=800)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32", "auto"], default="float16")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    p.add_argument("--low-cpu-mem-usage", action="store_true")
    return p.parse_args()


def import_runtime():
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Ascend inference needs PyTorch plus torch_npu. "
            "Install the CANN-matched torch/torch_npu packages first."
        ) from exc

    return torch


def resolve_dtype(torch, name: str):
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def set_npu_device(torch, device: str) -> None:
    if not device.startswith("npu"):
        raise ValueError("--device must be like npu:0")
    if not hasattr(torch, "npu"):
        raise RuntimeError("torch_npu is not available: torch has no npu backend")
    torch.npu.set_device(device)


def build_inputs(tokenizer, prompt: str, use_chat_template: bool) -> Dict[str, Any]:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer(rendered, return_tensors="pt")

    return tokenizer(prompt, return_tensors="pt", add_special_tokens=True)


def main() -> int:
    args = parse_args()
    torch = import_runtime()
    set_npu_device(torch, args.device)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = resolve_dtype(torch, args.dtype)
    print(f"[Ascend] device={args.device}, dtype={args.dtype}")
    print(f"[Ascend] loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[Ascend] loading model: {args.model}")
    load_kwargs = {
        "trust_remote_code": args.trust_remote_code,
    }
    if args.low_cpu_mem_usage:
        load_kwargs["low_cpu_mem_usage"] = True
    if dtype != "auto":
        load_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    model.to(args.device)

    inputs = build_inputs(
        tokenizer,
        args.prompt,
        use_chat_template=not args.no_chat_template,
    )
    input_len = int(inputs["input_ids"].shape[-1])
    if input_len >= args.max_seq:
        raise ValueError(f"input length {input_len} must be smaller than --max-seq {args.max_seq}")

    max_new_tokens = min(args.max_new_tokens, args.max_seq - input_len)
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    print("\n========== prompt ==========")
    print(args.prompt)
    print("\n========== input length ==========")
    print(input_len)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
        })
    else:
        gen_kwargs["do_sample"] = False

    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    elapsed = time.perf_counter() - t0

    gen_ids = output_ids[0, input_len:]
    gen_tokens = int(gen_ids.numel())
    text = tokenizer.decode(gen_ids, skip_special_tokens=True, errors="replace")

    print("\n========== generated text ==========")
    print(text)
    print("\n========== performance ==========")
    print(f"generated_tokens={gen_tokens}")
    print(f"elapsed_s={elapsed:.3f}")
    if elapsed > 0:
        print(f"tokens_per_s={gen_tokens / elapsed:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
