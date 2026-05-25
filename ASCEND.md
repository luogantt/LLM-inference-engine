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

## Direct AscendCL Engine Skeleton

For a CUDA-like path, this branch also provides a direct C ABI shared library:

```text
Python tokenizer
  -> ctypes
  -> build/libllm_ascend.so
  -> AscendCL / CANN runtime
  -> Ascend HBM
```

This is the first direct-runtime stage. It initializes AscendCL, scans safetensors metadata, allocates HBM, copies prefill token ids to the device, and verifies an H2D/D2H roundtrip. Transformer decode kernels are intentionally not faked yet.

Build it on the Ascend machine:

```bash
make -f Makefile.cuda_lib clean-lib

make -f Makefile.cuda_lib lib-ascend \
  ASCEND_HOME=/usr/local/Ascend/cann-8.5.1
```

If your image has the standard `latest` symlink, this also works:

```bash
make -f Makefile.cuda_lib lib-ascend
```

Smoke test the direct runtime and prefill path:

```bash
export ASCEND_VISIBLE_DEVICES=4
export ASCEND_DEVICE_ID=0

python python_infer.py \
  --model ./deepseek-r1-7b \
  --lib ./build/libllm_ascend.so \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 1 \
  --max-seq 800 \
  --tokenizer-backend tokenizers \
  --prefill-only
```

Expected log shape:

```text
[Python] backend: ascend-direct-acl
[Ascend][time] create engine ...
[Ascend][time] prefill copied token_ids to HBM ...
[Python] prefill-only finished
```

The `tokenizers` backend is intentional here. It avoids importing `transformers` and `torch_npu` in the same process as the direct AscendCL shared library, which keeps the direct runtime smoke test isolated.

## Load Weights Into Ascend HBM

The direct engine can now copy safetensors weights into Ascend HBM. This is controlled by `ASCEND_LOAD_WEIGHTS`:

```text
none     default, only initializes runtime and token buffer
minimal  embedding + final norm + optional lm_head
layer0   embedding + layer 0 attention/mlp/norm weights + final norm + optional lm_head
all      all safetensors tensors
```

Start with `minimal`:

```bash
export ASCEND_VISIBLE_DEVICES=4
export ASCEND_DEVICE_ID=0
export ASCEND_LOAD_WEIGHTS=minimal

python python_infer.py \
  --model ./deepseek-r1-7b \
  --lib ./build/libllm_ascend.so \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 1 \
  --max-seq 800 \
  --tokenizer-backend tokenizers \
  --prefill-only
```

Expected extra logs:

```text
[Ascend][time] weight loaded to HBM, name=model.embed_tokens.weight ...
[Ascend][time] requested weights loaded, mode=minimal ...
```

## Run RMSNorm Reference Stage

After embedding lookup succeeds, enable the RMSNorm reference stage:

```bash
export ASCEND_VISIBLE_DEVICES=4
export ASCEND_DEVICE_ID=0
export ASCEND_LOAD_WEIGHTS=minimal
export ASCEND_RUN_EMBED=1
export ASCEND_RUN_RMSNORM=1

python python_infer.py \
  --model ./deepseek-r1-7b \
  --lib ./build/libllm_ascend.so \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 1 \
  --max-seq 800 \
  --tokenizer-backend tokenizers \
  --prefill-only
```

Current RMSNorm is a correctness/reference stage:

```text
hidden HBM -> D2H -> host RMSNorm math -> H2D -> hidden HBM
```

It keeps the direct AscendCL data path clear while the next step is replacing the host math with an AscendC kernel.

Expected extra log:

```text
[Ascend][time] rmsnorm reference finished ...
```

## Run Layer0 Q Projection Reference

After RMSNorm succeeds, load layer 0 weights and run the q projection reference stage:

```bash
export ASCEND_VISIBLE_DEVICES=4
export ASCEND_DEVICE_ID=0
export ASCEND_LOAD_WEIGHTS=layer0
export ASCEND_RUN_EMBED=1
export ASCEND_RUN_RMSNORM=1
export ASCEND_RUN_QPROJ=1
export ASCEND_QPROJ_REF_TOKENS=1

python python_infer.py \
  --model ./deepseek-r1-7b \
  --lib ./build/libllm_ascend.so \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 1 \
  --max-seq 800 \
  --tokenizer-backend tokenizers \
  --prefill-only
```

Current q projection is also a correctness/reference stage:

```text
hidden HBM -> D2H -> host GEMV -> H2D -> Q HBM
```

By default only one token is computed because the reference GEMV is intentionally simple. Increase `ASCEND_QPROJ_REF_TOKENS` if needed.

Expected extra log:

```text
[Ascend][time] q_proj reference finished ...
```

Next direct-engine milestones:

1. Replace RMSNorm reference with an AscendC kernel.
2. Replace q_proj reference with an AscendC/GEMV kernel and add k/v/o projections.
3. Implement RoPE, KV Cache layout, GQA Attention, SwiGLU MLP, LM Head, and sampling.
