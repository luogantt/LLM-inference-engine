# Moore Threads MUSA Inference Notes

This branch adds a first-pass Moore Threads MUSA build path for the CUDA inference engine.

The verified target environment from AutoDL is:

```text
GPU: MTT S4000
Memory: 49152 MiB
Driver: 2.7.0
MUSA Toolkit: 3.1.0
mcc: /usr/local/musa/bin/mcc
```

This is not a Huawei Ascend environment and it is not NVIDIA CUDA. Moore Threads uses the MUSA runtime and the `mcc` compiler. This project keeps the existing CUDA-style kernel source, but the MUSA build now uses native MUSA headers and maps the small set of CUDA runtime calls used by this engine to MUSA runtime calls.

## Check The Device

```bash
mthreads-gmi
musa_version_query
ls /usr/local/musa
```

Expected signs of a usable device:

```text
Name: MTT S4000
Driver Version: 2.7.0
musa_toolkits version: 3.1.0
```

## Build

From the repository root:

```bash
export PATH=/usr/local/musa/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH

make -f Makefile.cuda_lib clean-lib
make -f Makefile.cuda_lib lib-musa
```

The output library is:

```text
build/libllm_musa.so
```

The new Makefile target uses:

```text
mcc -x musa -mtgpu -DUSE_MUSA=1
```

and links `libmusart`.

The first compatibility attempt used `-cuda_wrapper`, but the AutoDL MTT S4000 image does not provide NVIDIA CUDA headers such as `cuda.h`. In that environment, native MUSA compilation is the cleaner path:

```text
musa_runtime.h
musa_fp16.h
musaMalloc / musaMemcpy / musaMemset / musaDeviceSynchronize
```

## Run

Use the MUSA shared library with the existing Python entry point:

```bash
MUSA_VISIBLE_DEVICES=0 python python_infer.py \
  --model /path/to/deepseek-r1-7b \
  --lib ./build/libllm_musa.so \
  --prompt "你好 deepseek 介绍一下黑格尔的思想" \
  --max-new-tokens 128 \
  --max-seq 800
```

Watch the GPU in another terminal:

```bash
watch -n 1 mthreads-gmi
```

If memory usage and GPU utilization stay at zero, the program did not reach the MUSA device path.

## Important Limitations

This is a compatibility-first port, not a fully optimized MUSA backend.

Current limitations:

- The engine still loads HuggingFace safetensors model directories.
- It does not load GGUF files such as `DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf`.
- The compiled constants currently target DeepSeek-R1-Distill-Qwen-7B shape:

```text
N_LAYERS = 28
HIDDEN = 3584
N_HEADS = 28
N_KV_HEADS = 4
INTERMEDIATE = 18944
VOCAB_SIZE = 152064
```

- WMMA is disabled for MUSA builds because the CUDA `nvcuda::wmma` path is NVIDIA-specific.
- This path still uses the original custom kernels, with the required runtime calls mapped to native MUSA APIs.

For the existing 14B GGUF file, use llama.cpp with its MUSA backend instead:

```bash
cd ~/llama.cpp
cmake -B build-musa -DGGML_MUSA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-musa -j$(nproc)
./build-musa/bin/llama-bench -m ~/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf -ngl 999 -p 512 -n 128
```

## Troubleshooting

If `mcc` is not found:

```bash
export PATH=/usr/local/musa/bin:$PATH
```

If `libmusart.so` cannot be loaded:

```bash
export LD_LIBRARY_PATH=/usr/local/musa/lib:$LD_LIBRARY_PATH
```

If the build fails on CUDA-only headers, confirm that the MUSA target is being used:

```bash
make -f Makefile.cuda_lib lib-musa
```

Do not use the NVIDIA CUDA target on MTT S4000:

```bash
make -f Makefile.cuda_lib lib A=sm_80
```

That target is for NVIDIA GPUs and `nvcc`.
