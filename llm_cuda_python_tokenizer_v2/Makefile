NVCC ?= nvcc
A ?= sm_80
COMMON_FLAGS = -O2 -std=c++17 -lineinfo

.PHONY: lib clean

lib:
	mkdir -p build
	$(NVCC) $(COMMON_FLAGS) -arch=$(A) -Xcompiler -fPIC -shared src/llm_cuda_lib.cu -o build/libllm_cuda.so

clean:
	rm -rf build
