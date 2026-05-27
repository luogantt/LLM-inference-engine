# vLLM-Ascend 推理测试教程

本文用于在同一台昇腾机器上测试 `vllm-ascend`，并说明如何把自己的 Ascend 算子接入 vLLM 路径。

注意：本项目当前的 `build/libllm_ascend.so` 是独立 C++ 推理引擎，vLLM 不会自动调用它。要让 vLLM 推理“自己的算子”，需要把算子移植到 `vllm-ascend` 的自定义 aclnn 算子体系里，然后在 vLLM-Ascend 的模型执行路径中调用。

## 1. 环境检查

```bash
npu-smi info

python - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))
print("npu available:", torch.npu.is_available())
print("npu count:", torch.npu.device_count())
PY
```

如果你的机器使用 CANN 8.5.1，优先使用与当前 CANN、torch、torch-npu 匹配的 `vllm-ascend` 版本。下面以 `vllm==0.19.1`、`vllm-ascend==0.19.1rc1` 为例。

## 2. 安装 vllm-ascend

建议新建环境，避免影响本项目现有推理环境。

```bash
python -m venv ~/venvs/vllm-ascend
source ~/venvs/vllm-ascend/bin/activate

if [ -f /usr/local/Ascend/cann-8.5.1/set_env.sh ]; then
  source /usr/local/Ascend/cann-8.5.1/set_env.sh
elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

python -m pip install -U pip setuptools wheel

pip install vllm==0.19.1
pip install \
  --extra-index-url https://mirrors.huaweicloud.com/repository/pypi/simple \
  vllm-ascend==0.19.1rc1
```

验证安装：

```bash
python - <<'PY'
import vllm
import vllm_ascend
import torch
import torch_npu

print("vllm:", vllm.__version__)
print("vllm_ascend imported")
print("npu available:", torch.npu.is_available())
PY
```

## 3. 单卡 offline 推理测试

这个测试用于和本项目的 `python_infer.py --lib ./build/libllm_ascend.so` 做同 prompt、同 128 tokens 对比。

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

llm.generate([PROMPT], sampling)  # warmup

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

export ASCEND_VISIBLE_DEVICES=4
export ASCEND_RT_VISIBLE_DEVICES=4
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256

python vllm_ascend_offline_test.py 2>&1 | tee vllm_ascend_offline_128.log
```

如果 `dtype="float16"` 报错，可以改成：

```python
dtype="bfloat16"
```

或者：

```python
dtype="auto"
```

## 4. OpenAI API 服务推理

启动服务：

```bash
cd ~/LLM-inference-engine

export ASCEND_VISIBLE_DEVICES=4
export ASCEND_RT_VISIBLE_DEVICES=4
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256

vllm serve ./deepseek-r1-7b \
  --served-model-name deepseek-r1-7b \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 800 \
  --max-num-seqs 1 \
  --dtype float16 \
  --enforce-eager \
  --trust-remote-code
```

另开终端请求：

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1-7b",
    "prompt": "黑格尔的哲学思想可以概括为",
    "max_tokens": 128,
    "temperature": 0
  }' | python -m json.tool
```

服务模式适合测 API 行为和并发吞吐；如果要和本项目逐 token decode 日志对比，优先用第 3 节 offline 脚本。

## 5. torchrun 跑 vLLM offline

`torchrun` 更适合 offline tensor parallel / pipeline parallel 推理。多进程时使用 vLLM 的 `distributed_executor_backend="external_launcher"`，让 worker 由 `torchrun` 外部启动。

先写脚本：

```bash
cd ~/LLM-inference-engine

cat > torchrun_vllm_ascend_offline.py <<'PY'
import argparse
import os
import time

import torch.distributed as dist
from vllm import LLM, SamplingParams


def is_rank0():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="./deepseek-r1-7b")
parser.add_argument("--prompt", default="黑格尔的哲学思想可以概括为")
parser.add_argument("--max-model-len", type=int, default=800)
parser.add_argument("--max-tokens", type=int, default=128)
parser.add_argument("--tp-size", type=int, default=1)
parser.add_argument("--pp-size", type=int, default=1)
parser.add_argument("--dtype", default="float16")
parser.add_argument("--enforce-eager", action="store_true")
args = parser.parse_args()

world_size = int(os.environ.get("WORLD_SIZE", "1"))

llm_kwargs = dict(
    model=args.model,
    tokenizer=args.model,
    trust_remote_code=True,
    dtype=args.dtype,
    tensor_parallel_size=args.tp_size,
    pipeline_parallel_size=args.pp_size,
    max_model_len=args.max_model_len,
    max_num_seqs=1,
    gpu_memory_utilization=0.90,
    seed=1,
)

if args.enforce_eager:
    llm_kwargs["enforce_eager"] = True

if world_size > 1:
    llm_kwargs["distributed_executor_backend"] = "external_launcher"

llm = LLM(**llm_kwargs)
sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

llm.generate([args.prompt], sampling)  # warmup

t0 = time.perf_counter()
outputs = llm.generate([args.prompt], sampling)
t1 = time.perf_counter()

if is_rank0():
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

单卡 `torchrun`：

```bash
export ASCEND_VISIBLE_DEVICES=4
export ASCEND_RT_VISIBLE_DEVICES=4
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256

torchrun --nproc-per-node=1 torchrun_vllm_ascend_offline.py \
  --model ./deepseek-r1-7b \
  --max-model-len 800 \
  --max-tokens 128 \
  --tp-size 1 \
  --pp-size 1 \
  --enforce-eager \
  2>&1 | tee torchrun_vllm_ascend_1npu_128.log
```

双卡 tensor parallel 示例：

```bash
export ASCEND_VISIBLE_DEVICES=0,1
export ASCEND_RT_VISIBLE_DEVICES=0,1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256

torchrun --nproc-per-node=2 torchrun_vllm_ascend_offline.py \
  --model ./deepseek-r1-7b \
  --max-model-len 800 \
  --max-tokens 128 \
  --tp-size 2 \
  --pp-size 1 \
  --enforce-eager \
  2>&1 | tee torchrun_vllm_ascend_2npu_tp2_128.log
```

`--nproc-per-node` 应等于 `--tp-size * --pp-size`。如果只想和本项目单 batch decode 对比，先用单卡。

## 6. 如何让 vLLM 推理自己的算子

本项目当前路径：

```text
python_infer.py -> build/libllm_ascend.so -> AscendCL / ACLNN
```

vLLM-Ascend 路径：

```text
vLLM Python engine -> vllm-ascend plugin -> torch_npu / aclnn custom ops
```

所以不能直接把 `build/libllm_ascend.so` 交给 vLLM。要让 vLLM 使用你的算子，有两条路线。

### 路线 A：移植成 vllm-ascend 自定义 aclnn op

适合把你当前的 QKV fusion、MLP gate/up fusion、attention kernel 接入 vLLM。

大致步骤：

```bash
cd ~
git clone --depth 1 --branch v0.19.1rc1 https://github.com/vllm-project/vllm-ascend.git
cd ~/vllm-ascend
```

然后按照 vllm-ascend 的 custom aclnn op 目录结构新增算子，例如：

```text
csrc/
  custom_ops/
    my_qkv_fused/
      op_host/
      op_kernel/
      CMakeLists.txt
```

接着做三件事：

1. 在 C++/AscendC 侧实现你的算子 kernel 和 host launch。
2. 在 Python binding 中注册成可调用 op，例如 `torch.ops._C_ascend.my_qkv_fused(...)`。
3. 在 vllm-ascend 的模型执行路径中，把原来的 torch/torch_npu 算子替换成你的 op。

测试单个自定义 op：

```bash
python - <<'PY'
import torch
import torch_npu

# 示例：实际名字以你注册的 op 为准
# y = torch.ops._C_ascend.my_qkv_fused(x, w)
print("custom op smoke test placeholder")
PY
```

### 路线 B：保留本项目 direct .so，只把 vLLM 当 baseline

这是当前最稳的评测路线：

```text
torch_npu baseline       -> python_infer_ascend.py
vllm-ascend baseline     -> vllm_ascend_offline_test.py
your direct Ascend .so   -> python_infer.py --lib ./build/libllm_ascend.so
```

这样可以清楚证明：

```text
你的 direct .so > torch_npu
你的 direct .so > vllm-ascend
```

但这不代表 vLLM 已经调用了你的算子。只有完成路线 A，才能说“vLLM 推理正在使用我的算子”。

## 7. 推荐对比口径

统一使用：

```text
model: ./deepseek-r1-7b
prompt: 黑格尔的哲学思想可以概括为
max_new_tokens: 128
max_model_len / max_seq: 800
batch size: 1
temperature: 0
```

本项目 direct .so：

```bash
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

vLLM-Ascend：

```bash
python vllm_ascend_offline_test.py 2>&1 | tee vllm_ascend_offline_128.log
```

torchrun vLLM-Ascend：

```bash
torchrun --nproc-per-node=1 torchrun_vllm_ascend_offline.py \
  --model ./deepseek-r1-7b \
  --max-model-len 800 \
  --max-tokens 128 \
  --tp-size 1 \
  --pp-size 1 \
  --enforce-eager \
  2>&1 | tee torchrun_vllm_ascend_1npu_128.log
```

## 参考

- vLLM-Ascend installation: https://docs.vllm.ai/projects/ascend/en/main/installation.html
- vLLM-Ascend single NPU tutorial: https://vllm-ascend.readthedocs.io/en/v0.8.4rc1/tutorials/single_npu.html
- vLLM torchrun example: https://docs.vllm.ai/en/stable/examples/features/torchrun/
