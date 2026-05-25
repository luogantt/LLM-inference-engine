# Ascend 910 Inference

This branch adds a first Ascend path for running the DeepSeek safetensors model on Ascend 910 by using `torch_npu` and `transformers`.

The existing handwritten `src/llm_cuda_lib.cu` engine is CUDA-specific and cannot be compiled directly for Ascend. This script is the fastest verification path for the NPU machine. After the model runs correctly, the next deeper direction is an ACL / custom-operator backend.

## Check NPU

```bash
npu-smi info
```

Expected hardware is similar to:

```text
Name: Ascend910
HBM-Usage: ... / 65536 MB
Health: OK
```

## Download DeepSeek safetensors

For China mainland networks, use ModelScope:

```bash
pip install -U modelscope

python download_model.py \
  --source modelscope \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir /root/autodl-tmp/deepseek-r1-7b
```

## Install runtime

The Ascend Python runtime must match the CANN version installed on the pod. If `torch_npu` is already installed, skip this section.

Check first:

```bash
python - <<'PY'
import torch
import torch_npu
print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available())
PY
```

If it fails, install the CANN-matched PyTorch and torch_npu packages provided by your Ascend image or platform documentation.

## Run inference

If the pod exposes NPU 4 from `npu-smi`, set it as visible and use logical `npu:0` inside Python:

```bash
export ASCEND_VISIBLE_DEVICES=4

python python_infer_ascend.py \
  --model /root/autodl-tmp/deepseek-r1-7b \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 128 \
  --max-seq 800 \
  --device npu:0 \
  --dtype float16
```

If the runtime has already remapped the visible card to logical device 0, the same command works. If not, try `--device npu:4`.

## Notes

- `python_infer_ascend.py` is framework-based and uses `torch_npu`.
- The CUDA shared library path `--lib ./build/libllm_cuda.so` is not used on Ascend.
- For lower latency later, the next engineering step is to build an Ascend backend with ACL / custom operators, using the same separation idea as hardware plugin backends such as vLLM Ascend.
