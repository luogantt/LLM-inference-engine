import argparse
import ctypes
import os
from typing import List

from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/lg/推理/推理引擎/deepseek-r1-7b")
    p.add_argument("--lib", default="./build/libllm_cuda.so")
    p.add_argument("--prompt", default="你好 deepseek")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--max-seq", type=int, default=256)
    p.add_argument("--no-chat-template", action="store_true")
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

        self.lib.llm_last_error.argtypes = []
        self.lib.llm_last_error.restype = ctypes.c_char_p

        print(f"[Python] load lib: {lib_path}")
        print(f"[Python] model dir: {model_dir}")

        self.handle = self.lib.llm_create(model_dir.encode("utf-8"), max_seq)
        if not self.handle:
            raise RuntimeError(self.last_error())

    def last_error(self) -> str:
        p = self.lib.llm_last_error()
        return p.decode("utf-8", errors="replace") if p else "unknown error"

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

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.llm_destroy(self.handle)
            self.handle = None


def encode_prompt(tokenizer, prompt: str, use_chat_template: bool) -> List[int]:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return [int(x) for x in ids]

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
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

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

    engine = CudaLLM(args.lib, args.model, args.max_seq)

    print("\n========== CUDA prefill ==========")
    engine.prefill(input_ids)

    stop_ids = eos_set(tokenizer)
    gen_ids: List[int] = []

    print("\n========== CUDA decode ==========")
    for i in range(args.max_new_tokens):
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
