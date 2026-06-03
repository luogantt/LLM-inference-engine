# DeepSeek-V4-Flash 推理适配说明

本文记录 `cuda_A800_deepseekv4` 分支对 DeepSeek-V4-Flash 的第一阶段适配。

当前已接入的是 **DeepSeek-V4-Flash 官方 PyTorch/CUDA 推理路径**，通过根目录脚本 `python_infer_deepseek_v4_flash.py` 统一转换、加载和测速。项目原来的 `build/libllm_cuda.so` direct CUDA dense 引擎暂时还不能直接运行 DeepSeek-V4-Flash，因为两者模型结构差异很大。

## 为什么不能直接复用 7B CUDA 引擎

当前 `src/llm_cuda_lib.cu` 主要面向 DeepSeek-R1-Distill-Qwen-7B / Qwen dense 结构，核心假设包括：

```text
layers=28
hidden=3584
heads=28
kv_heads=4
head_dim=128
intermediate=18944
vocab=152064
weight names=model.layers.*.self_attn.q_proj / mlp.gate_proj ...
```

DeepSeek-V4-Flash 的配置是另一套架构：

```text
model_type=deepseek_v4
architecture=DeepseekV4ForCausalLM
total_params=284B
activated_params=13B
layers=43
hidden=4096
heads=64
kv_heads=1
head_dim=512
rope_head_dim=64
vocab=129280
n_routed_experts=256
n_shared_experts=1
num_experts_per_tok=6
moe_intermediate_size=2048
sliding_window=128
max_position_embeddings=1048576
precision=FP4 experts + FP8 mixed
```

权重名也不同，例如：

```text
layers.0.attn.wq_a.weight
layers.0.attn.wq_b.weight
layers.0.attn.wkv.weight
layers.0.attn.wo_a.weight
layers.0.attn.wo_b.weight
layers.0.ffn.experts.0.w1.weight
layers.0.ffn.shared_experts.w1.weight
```

所以它不是“改 config 就能跑”的模型，需要新增 MoE、FP8/FP4 GEMM、CSA/HCA、mHC、DeepSeek-V4 tokenizer encoding 等完整路径。

## 已新增入口

根目录新增：

```text
python_infer_deepseek_v4_flash.py
```

它复用以下官方文件：

```text
DeepSeek-V4-Flash/inference/model.py
DeepSeek-V4-Flash/inference/kernel.py
DeepSeek-V4-Flash/inference/generate.py
DeepSeek-V4-Flash/inference/convert.py
DeepSeek-V4-Flash/encoding/encoding_dsv4.py
```

这个入口支持：

1. 从 HuggingFace safetensors 转换成官方推理分片。
2. 使用 `torchrun` 多卡模型并行推理。
3. 用统一格式输出 `generated_tokens`、`elapsed_s`、`tokens_per_s`。
4. 支持 `chat` / `thinking` 两种 DeepSeek-V4 prompt encoding。

## 安装依赖

建议在 CUDA A800 机器上新建环境：

```bash
python -m venv ~/venvs/deepseek-v4-flash
source ~/venvs/deepseek-v4-flash/bin/activate

python -m pip install -U pip setuptools wheel

pip install -r DeepSeek-V4-Flash/inference/requirements.txt
```

如果 `tilelang` 或 `fast_hadamard_transform` 下载慢，可以换国内 PyPI 源。

## 下载模型

DeepSeek-V4-Flash 权重较大，`model.safetensors.index.json` 中标注总大小约 160GB。建议放在高速 NVMe 上。

ModelScope：

```bash
python download_model.py \
  --source modelscope \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --local-dir ./DeepSeek-V4-Flash
```

HuggingFace：

```bash
python download_model.py \
  --source huggingface \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --local-dir ./DeepSeek-V4-Flash
```

## 转换权重

官方推理代码要求先把 HF 权重转换成按 model parallel 切好的分片。

A800 单机 4 卡示例：

```bash
cd ~/LLM-inference-engine
source ~/venvs/deepseek-v4-flash/bin/activate

python python_infer_deepseek_v4_flash.py \
  --convert \
  --convert-only \
  --model ./DeepSeek-V4-Flash \
  --save-path ./DeepSeek-V4-Flash-converted \
  --model-parallel 4 \
  --n-experts 256
```

如果想把 expert 从 FP4 转成 FP8：

```bash
python python_infer_deepseek_v4_flash.py \
  --convert \
  --convert-only \
  --model ./DeepSeek-V4-Flash \
  --save-path ./DeepSeek-V4-Flash-converted-fp8 \
  --model-parallel 4 \
  --n-experts 256 \
  --expert-dtype fp8
```

转换后目录应包含：

```text
model0-mp4.safetensors
model1-mp4.safetensors
model2-mp4.safetensors
model3-mp4.safetensors
tokenizer.json
tokenizer_config.json
```

## 推理

A800 单机 4 卡：

```bash
cd ~/LLM-inference-engine
source ~/venvs/deepseek-v4-flash/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc-per-node 4 python_infer_deepseek_v4_flash.py \
  --ckpt-path ./DeepSeek-V4-Flash-converted \
  --config ./DeepSeek-V4-Flash/inference/config.json \
  --prompt "黑格尔的哲学思想可以概括为" \
  --max-new-tokens 128 \
  --max-seq-len 4096 \
  --max-batch-size 1 \
  --temperature 0 \
  --warmup \
  2>&1 | tee deepseek_v4_flash_a800_4gpu_128.log
```

输出格式：

```text
========== generated text ==========
...

========== performance ==========
generated_tokens=...
elapsed_s=...
tokens_per_s=...
```

## Batch 输入

输入文件用空行分隔 prompt：

```bash
cat > prompts.txt <<'EOF'
黑格尔的哲学思想可以概括为

请用一段话解释 CUDA kernel fusion 的意义
EOF
```

运行：

```bash
torchrun --nproc-per-node 4 python_infer_deepseek_v4_flash.py \
  --ckpt-path ./DeepSeek-V4-Flash-converted \
  --config ./DeepSeek-V4-Flash/inference/config.json \
  --input-file prompts.txt \
  --max-new-tokens 128 \
  --max-seq-len 4096 \
  --max-batch-size 2 \
  --temperature 0 \
  --warmup \
  2>&1 | tee deepseek_v4_flash_batch.log
```

## 后续 direct CUDA 引擎适配路线

如果要让 `build/libllm_cuda.so` 直接跑 DeepSeek-V4-Flash，需要分阶段做：

1. 配置解析：支持 `model_type=deepseek_v4`、43 层、4096 hidden、129280 vocab。
2. 权重加载：支持 V4 权重名和 46 个 safetensors shard。
3. FP8/FP4 解码：支持 e4m3 FP8、e2m1 FP4 packed expert 权重和 scale。
4. GEMM：实现 FP8 GEMM、FP4 expert GEMM、block scale 反量化。
5. Attention：实现 DeepSeek-V4 的 compressed sparse attention / heavily compressed attention。
6. MoE：实现 router、top-k expert 选择、shared expert、256 routed experts、6 activated experts。
7. mHC：实现 hyper-connection 相关张量和残差路径。
8. KV cache：按 sliding window 与 1M context 设计压缩 KV/cache 管理。
9. 多 GPU：支持 model parallel 分片加载、跨卡 all-reduce / gather。
10. Tokenizer：接入 `encoding_dsv4.py` 的 chat / thinking prompt 格式。

第一阶段先用官方推理路径跑通 A800 多卡 baseline；第二阶段再逐个把热路径迁移到自研 CUDA kernel。

## A800 FP8/FP4 fallback

DeepSeek-V4-Flash 官方 TileLang FP8 GEMM 在 A800(sm80) 上会触发 SM89 FP8 MMA 路径断言，因此不能直接运行官方 Flash kernel：

```text
Attempting to use SM89_16x8x32_F32E4M3E4M3F32_TN without CUTE_ARCH_MMA_F32_SM89_ENABLED
RuntimeError: CUDALaunch CUDA_ERROR_ASSERT
```

A800 需要单独实现 BF16/FP16 fallback 或自研 CUDA kernel。本分支新增一个保守兼容开关：

```bash
export A800_FORCE_DEQUANT_GEMM=1
export A800_DEQUANT_DTYPE=bf16
```

开启后，`DeepSeek-V4-Flash/inference/model.py` 的 `linear()` 会绕过 TileLang `fp8_gemm/fp4_gemm`：

```text
FP8 dense linear: FP8 weight + block scale -> BF16/FP16 dequant -> F.linear/cuBLAS
FP4 expert linear: FP4 packed weight + per-32 scale -> BF16/FP16 dequant -> F.linear/cuBLAS
HC split/sinkhorn: TileLang kernel -> PyTorch fallback
Sparse attention: TileLang kernel -> PyTorch fallback
Activation quant simulation: skipped by default on A800 fallback
Hadamard rotation: skipped by default on A800 fallback to avoid requiring fast_hadamard_transform
```

这个路径的目标是先让 A800 跑通 DeepSeek-V4-Flash，不追求官方 Flash kernel 的速度。真正要快，需要把 FP8/FP4 unpack、scale 和 GEMM 融合成 A800(sm80) 专用 CUDA kernel。

最小验证命令：

```bash
cd /data3/ledi/deepseekv4_engin/LLM-inference-engine

export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.0"
export A800_FORCE_DEQUANT_GEMM=1
export A800_DEQUANT_DTYPE=bf16

CUDA_VISIBLE_DEVICES=2,3,4,5 python -m torch.distributed.run \
  --standalone \
  --nproc-per-node 4 \
  python_infer_deepseek_v4_flash.py \
  --ckpt-path ../models/DeepSeek-V4-Flash-converted \
  --config ../models/DeepSeek-V4-Flash/inference/config_a800_fp32scale.json \
  --prompt "hello" \
  --max-new-tokens 1 \
  --max-seq-len 4096 \
  --max-batch-size 1 \
  --temperature 0 \
  2>&1 | tee deepseek_v4_flash_a800_fallback_1tok.log
```

如果需要缓存反量化后的权重来减少重复开销，可以额外开启：

```bash
export A800_DEQUANT_CACHE=1
```

注意：这个缓存只缓存 dense FP8 权重，不缓存 FP4 expert 权重。FP4 expert 数量多，缓存成 BF16/FP16 后很容易把 A800 80GB 显存打满。

如果明确想尝试缓存 FP4 expert 权重，可以额外开启。该缓存是 LRU，有显存上限，默认 4096MB：

```bash
export A800_DEQUANT_CACHE_FP4=1
export A800_DEQUANT_CACHE_FP4_MB=4096
```

如果还有空闲显存，可以尝试 8192；如果出现 OOM，降到 2048 或直接关闭。先用 `--max-new-tokens 1` 验证通过，再做 128 token 性能测试。

如果需要保留 attention/indexer 里的 activation quant simulation，可以额外开启：

```bash
export A800_KEEP_ACT_QUANT=1
```

默认不建议开启，因为它会重新进入 TileLang FP8/FP4 quant kernel。

如果已经装好了 `fast_hadamard_transform`，并且需要保留原始 Hadamard rotation 路径，可以额外开启：

```bash
export A800_KEEP_ROTATE=1
```

A800 fallback 默认跳过这一步，目的是先绕开额外编译依赖，把 DeepSeek-V4-Flash 的推理链路跑通。
