import os
import json
import sys
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs
current_dir = os.path.dirname(os.path.abspath(__file__))
encoding_dir = os.path.join(current_dir, '../encoding')
sys.path.insert(0, os.path.abspath(encoding_dir))
from encoding_dsv4 import encode_messages, parse_message_from_completion_text


def _eos_check_interval() -> int:
    value = os.getenv("A800_EOS_CHECK_INTERVAL", "").strip()
    if value == "":
        return 1
    try:
        return max(0, int(value))
    except ValueError:
        return 1


def _single_prompt_fast_generate() -> bool:
    return os.getenv("A800_SINGLE_PROMPT_FAST_GENERATE", "").strip().lower() in {"1", "true", "yes", "on"}


def _distributed_argmax() -> bool:
    return os.getenv("A800_DISTRIBUTED_ARGMAX", "").strip().lower() in {"1", "true", "yes", "on"}


def _greedy_next_token(model: Transformer, input_ids: torch.Tensor, start_pos: int, use_distributed_argmax: bool) -> torch.Tensor:
    if use_distributed_argmax:
        return model.forward_argmax(input_ids, start_pos)
    return model.forward(input_ids, start_pos).argmax(dim=-1)


def sample(logits, temperature: float = 1.0):
    """Gumbel-max trick: equivalent to multinomial sampling but faster on GPU,
    since it avoids the GPU-to-CPU sync in torch.multinomial."""
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


def finalize_completion_tokens(tokens: torch.Tensor, prompt_lens: List[int], max_new_tokens: int, eos_id: int) -> List[List[int]]:
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        toks.append(eos_id)
        completion_tokens.append(toks)
    return completion_tokens


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    return_tensor: bool = False,
):
    """Batch generation with left-padded prompts.

    The first forward pass processes [min_prompt_len:] tokens (prefill phase).
    Subsequent passes generate one token at a time (decode phase). For positions
    still within a prompt, the ground-truth token overrides the model's prediction.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long)
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    use_distributed_argmax = temperature <= 0 and _distributed_argmax() and hasattr(model, "forward_argmax")
    if _single_prompt_fast_generate() and len(prompt_tokens) == 1:
        prev_pos = 0
        prompt_len = prompt_lens[0]
        eos_check_interval = _eos_check_interval()
        decode_steps = 0
        for cur_pos in range(prompt_len, total_len):
            if temperature > 0:
                logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
                next_token = sample(logits, temperature)
            else:
                next_token = _greedy_next_token(model, tokens[:, prev_pos:cur_pos], prev_pos, use_distributed_argmax)
            tokens[:, cur_pos] = next_token
            decode_steps += 1
            prev_pos = cur_pos
            if eos_check_interval and decode_steps % eos_check_interval == 0 and (next_token == eos_id).all():
                break
        if return_tensor:
            return tokens, prompt_lens
        return finalize_completion_tokens(tokens, prompt_lens, max_new_tokens, eos_id)

    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens))
    prompt_mask = tokens != -1
    eos_check_interval = _eos_check_interval()
    decode_steps = 0
    for cur_pos in range(min(prompt_lens), total_len):
        if temperature > 0:
            logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            next_token = sample(logits, temperature)
        else:
            next_token = _greedy_next_token(model, tokens[:, prev_pos:cur_pos], prev_pos, use_distributed_argmax)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        decode_steps += 1
        prev_pos = cur_pos
        if eos_check_interval:
            finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
            if decode_steps % eos_check_interval == 0 and finished.all():
                break
    if return_tensor:
        return tokens, prompt_lens
    return finalize_completion_tokens(tokens, prompt_lens, max_new_tokens, eos_id)


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    if interactive:
        args.max_batch_size = 1
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    print("load model")
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"), strict=False)
    torch.set_default_device("cuda")
    print("I'm DeepSeek 👋")

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.encode(encode_messages(messages, thinking_mode="chat"))
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0])
            print(completion)
            messages.append(parse_message_from_completion_text(completion, thinking_mode="chat"))
    else:
        with open(input_file) as f:
            prompts = f.read().split("\n\n")
        prompt_tokens = [tokenizer.encode(encode_messages([{"role": "user", "content": prompt}], thinking_mode="chat")) for prompt in prompts]
        completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id, temperature)
        completions = tokenizer.batch_decode(completion_tokens)
        for prompt, completion in zip(prompts, completions):
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)
