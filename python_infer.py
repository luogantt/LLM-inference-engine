import argparse
import ctypes
import json
import os
from typing import List


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/lg/推理/推理引擎/deepseek-r1-7b")
    p.add_argument("--lib", default="./build/libllm_cuda.so")
    p.add_argument("--prompt", default="你好 deepseek")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--max-seq", type=int, default=256)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--prefill-only", action="store_true")
    p.add_argument(
        "--tokenizer-backend",
        choices=["auto", "transformers", "tokenizers"],
        default="auto",
        help="use tokenizers for direct AscendCL smoke tests to avoid importing torch_npu",
    )
    return p.parse_args()


class CudaLLM:
    def __init__(self, lib_path: str, model_dir: str, max_seq: int):
        lib_path = os.path.abspath(lib_path)
        model_dir = os.path.abspath(model_dir)

        self.lib = ctypes.CDLL(lib_path)

        self.lib.llm_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.llm_create.restype = ctypes.c_void_p

        self.lib.llm_destroy.argtypes = [ctypes.c_void_p]
        self.lib.llm_destroy.restype = None

        self.lib.llm_prefill.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self.lib.llm_prefill.restype = ctypes.c_int

        self.lib.llm_decode_one.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.llm_decode_one.restype = ctypes.c_int

        self.lib.llm_set_repetition_penalty.argtypes = [
            ctypes.c_void_p,
            ctypes.c_float,
        ]
        self.lib.llm_set_repetition_penalty.restype = ctypes.c_int

        self.lib.llm_last_error.argtypes = []
        self.lib.llm_last_error.restype = ctypes.c_char_p

        try:
            self.lib.llm_backend_name.argtypes = []
            self.lib.llm_backend_name.restype = ctypes.c_char_p
        except AttributeError:
            pass

        print(f"[Python] load lib: {lib_path}")
        print(f"[Python] model dir: {model_dir}")

        self.handle = self.lib.llm_create(model_dir.encode("utf-8"), max_seq)
        if not self.handle:
            raise RuntimeError(self.last_error())

    def last_error(self) -> str:
        p = self.lib.llm_last_error()
        return p.decode("utf-8", errors="replace") if p else "unknown error"

    def backend_name(self) -> str:
        try:
            p = self.lib.llm_backend_name()
            if p:
                return p.decode("utf-8", errors="replace")
        except AttributeError:
            pass
        return "cuda"

    def check(self, ret: int):
        if ret != 0:
            raise RuntimeError(self.last_error())

    def prefill(self, ids: List[int]):
        arr_t = ctypes.c_int * len(ids)
        arr = arr_t(*[int(x) for x in ids])
        self.check(self.lib.llm_prefill(self.handle, arr, len(ids)))

    def decode_one(self) -> int:
        out = ctypes.c_int(-1)
        self.check(self.lib.llm_decode_one(self.handle, ctypes.byref(out)))
        return int(out.value)

    def set_repetition_penalty(self, penalty: float):
        self.check(self.lib.llm_set_repetition_penalty(self.handle, ctypes.c_float(penalty)))

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.llm_destroy(self.handle)
            self.handle = None


class TokenizersWrapper:
    def __init__(self, model_dir: str):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "tokenizers is required for --tokenizer-backend tokenizers. "
                "Install it with: pip install -U tokenizers"
            ) from exc

        tokenizer_path = os.path.join(model_dir, "tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.eos_token_id = self._find_token_id([
            "<|endoftext|>",
            "<|im_end|>",
            "<｜end▁of▁sentence｜>",
        ])
        self.chat_template = None
        config_path = os.path.join(model_dir, "tokenizer_config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self.chat_template = cfg.get("chat_template")
        except Exception:
            self.chat_template = None

    def _find_token_id(self, tokens):
        for token in tokens:
            tid = self.tokenizer.token_to_id(token)
            if isinstance(tid, int) and tid >= 0:
                return tid
        return None

    def encode(self, text: str, add_special_tokens: bool = True):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens: bool = True, errors: str = "replace"):
        del errors
        return self.tokenizer.decode([int(x) for x in ids], skip_special_tokens=skip_special_tokens)

    def convert_tokens_to_ids(self, token: str):
        tid = self.tokenizer.token_to_id(token)
        return tid if tid is not None else -1


def load_tokenizer(model_dir: str, backend: str, lib_path: str):
    if backend == "auto":
        lib_name = os.path.basename(lib_path).lower()
        backend = "tokenizers" if "ascend" in lib_name else "transformers"

    if backend == "tokenizers":
        print("[Python] tokenizer backend: tokenizers")
        return TokenizersWrapper(model_dir)

    print("[Python] tokenizer backend: transformers")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)


def encode_prompt(tokenizer, prompt: str, use_chat_template: bool) -> List[int]:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            return [int(x) for x in ids]
        print("[Python] chat template skipped: tokenizers backend does not render templates")

    return [int(x) for x in tokenizer.encode(prompt, add_special_tokens=True)]


def eos_set(tokenizer):
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    for s in ["<|endoftext|>", "<|im_end|>", "<｜end▁of▁sentence｜>"]:
        try:
            x = tokenizer.convert_tokens_to_ids(s)
            if isinstance(x, int) and x >= 0:
                ids.add(x)
        except Exception:
            pass
    return ids


def main():
    args = parse_args()

    print("[Python] loading tokenizer...")
    tokenizer = load_tokenizer(args.model, args.tokenizer_backend, args.lib)

    input_ids = encode_prompt(
        tokenizer,
        args.prompt,
        use_chat_template=not args.no_chat_template,
    )

    print("\n========== prompt ==========")
    print(args.prompt)
    print("\n========== input ids ==========")
    print(input_ids)
    print("input length:", len(input_ids))

    if len(input_ids) >= args.max_seq:
        raise ValueError(
            f"input length {len(input_ids)} must be smaller than --max-seq {args.max_seq}"
        )

    max_decode_tokens = min(args.max_new_tokens, args.max_seq - len(input_ids))
    if max_decode_tokens < args.max_new_tokens:
        print(
            "[Python] max-new-tokens clipped from "
            f"{args.max_new_tokens} to {max_decode_tokens} because max_seq={args.max_seq}"
        )

    engine = CudaLLM(args.lib, args.model, args.max_seq)
    backend = engine.backend_name()
    print(f"[Python] backend: {backend}")
    engine.set_repetition_penalty(args.repetition_penalty)
    print(f"[Python] repetition penalty: {args.repetition_penalty}")

    print(f"\n========== {backend} prefill ==========")
    engine.prefill(input_ids)
    if args.prefill_only:
        print("[Python] prefill-only finished")
        return

    stop_ids = eos_set(tokenizer)
    gen_ids: List[int] = []

    print(f"\n========== {backend} decode ==========")
    for i in range(max_decode_tokens):
        tid = engine.decode_one()
        gen_ids.append(tid)

        text = tokenizer.decode(gen_ids, skip_special_tokens=True, errors="replace")
        print(f"[{i}] token={tid}, text_so_far={repr(text)}", flush=True)

        if tid in stop_ids:
            print("[Python] hit EOS")
            break

    print("\n========== generated ids ==========")
    print(gen_ids)
    print("\n========== generated text ==========")
    print(tokenizer.decode(gen_ids, skip_special_tokens=True, errors="replace"))


if __name__ == "__main__":
    main()
