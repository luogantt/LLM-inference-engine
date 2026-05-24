#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_DIR = "/root/autodl-tmp/deepseek-r1-7b"


def run(cmd: list[str]) -> None:
    print("[download]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_modelscope() -> None:
    if shutil.which("modelscope"):
        return
    raise RuntimeError(
        "modelscope CLI was not found. Install it first:\n"
        "  pip install -U modelscope"
    )


def download_modelscope(model: str, local_dir: str) -> None:
    ensure_modelscope()
    run([
        "modelscope",
        "download",
        "--model",
        model,
        "--local_dir",
        local_dir,
    ])


def download_huggingface(model: str, local_dir: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it first:\n"
            "  pip install -U huggingface_hub"
        ) from exc

    snapshot_download(
        repo_id=model,
        local_dir=local_dir,
        resume_download=True,
    )


def check_files(local_dir: str) -> None:
    root = Path(local_dir)
    required = [
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError(
            "download finished, but required files are missing: "
            + ", ".join(missing)
        )

    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise RuntimeError("download finished, but no *.safetensors shards were found")

    print("[download] model directory:", root)
    print("[download] safetensors shards:", len(shards))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a safetensors model for this engine.")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    p.add_argument("--local-dir", default=DEFAULT_DIR, help=f"default: {DEFAULT_DIR}")
    p.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default="modelscope",
        help="model download source, default: modelscope",
    )
    p.add_argument(
        "--skip-check",
        action="store_true",
        help="skip required-file validation after download",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    Path(args.local_dir).mkdir(parents=True, exist_ok=True)

    try:
        if args.source == "modelscope":
            download_modelscope(args.model, args.local_dir)
        else:
            download_huggingface(args.model, args.local_dir)

        if not args.skip_check:
            check_files(args.local_dir)
        return 0
    except Exception as exc:
        print(f"[download][error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
