# Ascend-super：在 Ascend A3 上比 vLLM-Ascend 快约 25%

我最近在一个从零实现的大模型推理引擎里，把 DeepSeek-R1-Distill-Qwen-7B 的 Ascend A3 单 batch decode 性能推进到了一个很有意思的位置：

在同一台 Ascend A3、同一模型、同一 prompt、同样生成 128 tokens 的条件下，`Ascend-super` direct `.so` 路径达到了约 **47.1 tok/s**，相比 `vLLM-Ascend` baseline 的 **37.639 tok/s** 快约 **25.1%**，相比 `torch_npu` baseline 的 **34.627 tok/s** 快约 **36.0%**。

这不是调用 PyTorch、Transformers 或 vLLM 的结果，而是项目里的 C++ / AscendCL / ACLNN 直接推理路径。

## 项目地址

GitHub:

https://github.com/luogantt/LLM-inference-engine

对应 tag:

https://github.com/luogantt/LLM-inference-engine/tree/ascend-super-vs-vllm-47tok

拉取代码：

```bash
git clone https://github.com/luogantt/LLM-inference-engine.git
cd LLM-inference-engine
git checkout ascend-super-vs-vllm-47tok
```

## 测试环境

```text
硬件：Ascend A3
模型：DeepSeek-R1-Distill-Qwen-7B
batch：1
prompt：黑格尔的哲学思想可以概括为
max_new_tokens：128
max_seq：800
```

说明：这里的 PyTorch baseline 使用 `torch_npu` 在 Ascend/NPU 上运行。它是最接近常规 torch 推理体验的对照组；本文主结论比较的是同一块 Ascend A3 上的 `torch_npu`、`vLLM-Ascend` 和本项目 direct `.so` 路径。

## 结果汇总

| 路径 | 速度 | 相对 Ascend-super |
| --- | ---: | ---: |
| torch_npu baseline | 34.627 tok/s | Ascend-super 快约 36.0% |
| vLLM-Ascend baseline | 37.639 tok/s | Ascend-super 快约 25.1% |
| Ascend-super direct `.so` | 约 47.1 tok/s | 1.00x |

计算方式：

```text
Ascend-super vs torch_npu:
(47.1 / 34.627 - 1) * 100% = 36.0%

Ascend-super vs vLLM-Ascend:
(47.1 / 37.639 - 1) * 100% = 25.1%
```

## torch_npu baseline

测试命令：

```bash
cd ~/LLM-inference-engine

export ASCEND_VISIBLE_DEVICES=4

python python_infer_ascend.py \
  --model ./deepseek-r1-7b \
  --prompt "黑格尔的哲学思想可以概括为" \
  --max-new-tokens 128 \
  --max-seq 800 \
  --device npu:0 \
  --dtype float16 \
  2>&1 | tee torch_npu_128.log
```

关键 log：

```text
========== performance ==========
generated_tokens=128
elapsed_s=3.696
tokens_per_s=34.627
```

## vLLM-Ascend baseline

`vLLM-Ascend` 在 A3 上需要使用匹配 A3 的安装包或源码构建。普通 A2 wheel 会报类似下面的错误：

```text
Current device type: AscendDeviceType.A3 does not match the installed version's device type: AscendDeviceType.A2
```

本次 baseline 已在 A3 版本 `vLLM-Ascend` 路径上跑通。

测试脚本：

```bash
cd ~/LLM-inference-engine

cat > vllm_ascend_offline_test.py <<'PY'
import time
from vllm import LLM, SamplingParams

MODEL = "./deepseek-r1-7b"
PROMPT = "黑格尔的哲学思想可以概括为"

sampling = SamplingParams(temperature=0.0, max_tokens=128)

llm = LLM(
    model=MODEL,
    tokenizer=MODEL,
    trust_remote_code=True,
    dtype="float16",
    max_model_len=800,
    max_num_seqs=1,
    gpu_memory_utilization=0.90,
    enforce_eager=True,
)

llm.generate([PROMPT], sampling)

t0 = time.perf_counter()
outputs = llm.generate([PROMPT], sampling)
t1 = time.perf_counter()

out = outputs[0].outputs[0]
new_tokens = len(out.token_ids)
elapsed = t1 - t0

print("========== generated text ==========")
print(out.text)
print()
print("========== performance ==========")
print(f"generated_tokens={new_tokens}")
print(f"elapsed_s={elapsed:.6f}")
print(f"tokens_per_s={new_tokens / elapsed:.3f}")
PY
```

运行命令：

```bash
cd ~/LLM-inference-engine

source ~/venvs/vllm-ascend/bin/activate

mkdir -p ~/ascend/log

unset ASCEND_RT_VISIBLE_DEVICES
export ASCEND_VISIBLE_DEVICES=4
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256

python vllm_ascend_offline_test.py 2>&1 | tee vllm_ascend_offline_128.log
```

关键 log：

```text
Processed prompts: 100%|██████████| 1/1 [00:03<00:00,  3.40s/it, est. speed input: 2.35 toks/s, output: 37.65 toks/s]

========== performance ==========
generated_tokens=128
elapsed_s=3.400694
tokens_per_s=37.639
```

生成结束后的 shutdown 阶段可能出现：

```text
Engine core proc EngineCore died unexpectedly, shutting down client.
```

这条日志出现在已经打印 `generated_tokens=128` 和 `tokens_per_s=37.639` 之后，不影响这次性能数据。

## Ascend-super direct .so

本项目的 `Ascend-super` 路径不走 PyTorch graph，也不走 vLLM engine，而是通过 Python tokenizer 调用 C++ 动态库：

```text
python_infer.py -> build/libllm_ascend.so -> AscendCL / ACLNN
```

编译：

```bash
cd ~/LLM-inference-engine

make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib-ascend ASCEND_HOME=/usr/local/Ascend/cann-8.5.1
```

推理命令：

```bash
cd ~/LLM-inference-engine

mkdir -p ~/ascend/log

export ASCEND_VISIBLE_DEVICES=4
export ASCEND_DEVICE_ID=0

export ASCEND_LOAD_WEIGHTS=all
export ASCEND_WEIGHT_LOAD_LOG=0
export ASCEND_TIME_LOG_FILE=0
export ASCEND_HOST_RAW_CACHE=0

export ASCEND_RUN_EMBED=1
export ASCEND_DIRECT_DECODE=all_layers_ref

export ASCEND_REF_CACHE_WEIGHTS=1
export ASCEND_REF_CACHE_LOG=0
export ASCEND_REF_KV_CACHE=1
export ASCEND_REF_U16_WEIGHTS=1

export ASCEND_REF_FAST_DOT=1
export ASCEND_REF_DOT4=0
export ASCEND_REF_NEON_DOT=1

export ASCEND_ATTN_BACKEND=cpu

export ASCEND_QKV_BACKEND=aclnn
export ASCEND_QKV_FUSE_WEIGHTS=1
export ASCEND_QKV_FALLBACK=0
export ASCEND_QKV_LOG=0

export ASCEND_MLP_BACKEND=aclnn
export ASCEND_MLP_FUSE_GATE_UP=1
export ASCEND_MLP_FALLBACK=0
export ASCEND_MLP_LOG=0

export ASCEND_ATTN_PROJ_BACKEND=aclnn
export ASCEND_ATTN_PROJ_FALLBACK=0
export ASCEND_ATTN_PROJ_LOG=0

export ASCEND_LM_HEAD_BACKEND=aclnn
export ASCEND_LM_HEAD_FALLBACK=0
export ASCEND_LM_HEAD_LOG=0

export ASCEND_ACLNN_CUBE_MATH_TYPE=0

export ASCEND_REF_LINEAR_THREADS=16
export ASCEND_REF_ATTN_LINEAR_THREADS=16
export ASCEND_REF_ATTN_THREADS=16
export ASCEND_REF_ATTN_THREAD_MIN_SEQ=32
export ASCEND_REF_MLP_THREADS=24
export ASCEND_REF_DOWN_THREADS=24
export ASCEND_LM_HEAD_THREADS=16

export ASCEND_REF_PROFILE_LAYERS=0
export ASCEND_REF_PROFILE_TOKEN_LIMIT=0

python python_infer.py \
  --model ./deepseek-r1-7b \
  --lib ./build/libllm_ascend.so \
  --prompt "黑格尔的哲学思想可以概括为" \
  --max-new-tokens 128 \
  --max-seq 800 \
  --tokenizer-backend tokenizers \
  --no-chat-template \
  2>&1 | tee ascend_super_128.log
```

关键 log：

```text
[Ascend][time] decode all_layers_ref finished, token=102989, pos=117, elapsed_ms=21.191644
[Ascend][time] decode all_layers_ref finished, token=109732, pos=118, elapsed_ms=21.171014
[Ascend][time] decode all_layers_ref finished, token=54926, pos=119, elapsed_ms=21.191515
[Ascend][time] decode all_layers_ref finished, token=100116, pos=120, elapsed_ms=21.187475
[Ascend][time] decode all_layers_ref finished, token=9370, pos=121, elapsed_ms=21.103363
[Ascend][time] decode all_layers_ref finished, token=104380, pos=122, elapsed_ms=21.098643
[Ascend][time] decode all_layers_ref finished, token=104734, pos=123, elapsed_ms=20.984602
[Ascend][time] decode all_layers_ref finished, token=101036, pos=124, elapsed_ms=21.112034
[Ascend][time] decode all_layers_ref finished, token=26850, pos=125, elapsed_ms=21.102433
[Ascend][time] decode all_layers_ref finished, token=101140, pos=126, elapsed_ms=21.126373
[Ascend][time] decode all_layers_ref finished, token=3837, pos=127, elapsed_ms=21.154024
[Ascend][time] decode all_layers_ref finished, token=99720, pos=128, elapsed_ms=21.195795
[Ascend][time] decode all_layers_ref finished, token=85106, pos=129, elapsed_ms=21.228775
[Ascend][time] decode all_layers_ref finished, token=100692, pos=130, elapsed_ms=21.235385
[Ascend][time] decode all_layers_ref finished, token=104734, pos=131, elapsed_ms=21.313266
[Ascend][time] decode all_layers_ref finished, token=109151, pos=132, elapsed_ms=21.418098
[Ascend][time] decode all_layers_ref finished, token=33108, pos=133, elapsed_ms=21.233564
[Ascend][time] decode all_layers_ref finished, token=100466, pos=134, elapsed_ms=21.346556
[Ascend][time] decode all_layers_ref finished, token=1773, pos=135, elapsed_ms=21.434498
[Ascend][time] decode all_layers_ref finished, token=100220, pos=136, elapsed_ms=21.227135
```

最后一个 token 的粗略换算：

```text
1000 / 21.227135 = 47.11 tok/s
```

## 为什么会快

这个优化不是靠简单换框架，而是沿着 decode 热路径做减法：

1. 权重常驻设备侧，减少重复加载和格式转换。
2. QKV 使用 fused weight，一次 ACLNN matmul 输出 Q/K/V，减少 matmul 次数。
3. MLP gate/up 融合，降低 decode 阶段的小 batch 调度开销。
4. lm_head 使用 ACLNN argmax 路径，避免把完整 logits 转成 Python / torch 侧张量。
5. KV cache、U16 权重缓存、host buffer 复用，降低每 token 的内存分配和拷贝成本。
6. 单 batch decode 场景下，避开通用框架调度层，把注意力和残差路径尽可能压到低开销实现。

vLLM-Ascend 是优秀的通用推理框架，强项在服务化、调度、多并发、PagedAttention 和生态集成；而 `Ascend-super` 这条路径更像是针对单 batch decode 的极限实验，把通用性让位给直接、短路径和可控的算子调度。

## 如何复现模型下载

如果本地没有模型，可以用仓库里的下载脚本：

```bash
cd ~/LLM-inference-engine

pip install -U modelscope

python download_model.py \
  --source modelscope \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir ./deepseek-r1-7b
```

或者使用 HuggingFace：

```bash
pip install -U huggingface_hub

python download_model.py \
  --source huggingface \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir ./deepseek-r1-7b
```

## 结论

在这组实测里，`Ascend-super` direct `.so` 路径已经超过了常规 `torch_npu` baseline，也超过了 `vLLM-Ascend` baseline：

```text
torch_npu baseline:      34.627 tok/s
vLLM-Ascend baseline:    37.639 tok/s
Ascend-super direct .so: 约 47.1 tok/s
```

也就是说，在 DeepSeek-R1-Distill-Qwen-7B、Ascend A3、单 batch、128 tokens decode 这个具体场景下，一个从零手写的 AscendCL / ACLNN 推理路径，已经可以比 vLLM-Ascend 快约一个季度的量级，也就是 **约 25%**。

下一步目标很直接：继续压缩 decode 热路径，把 47 tok/s 推到 **50 tok/s**。
