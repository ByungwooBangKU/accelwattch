# RTX 3090 = compute capability 8.6. Override with `make SM=80` etc.
SM       ?= 86
NVCC     ?= nvcc
CUDA_DIR ?= $(dir $(shell which $(NVCC)))..

NVCCFLAGS = -O3 -std=c++17 -arch=sm_$(SM) -lineinfo -Xcompiler -Wall
LIBS      = -lnvToolsExt

TARGET = dram_util
SRC    = dram_util.cu

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LIBS)

clean:
	rm -f $(TARGET) *.nsys-rep *.sqlite *.qdrep

.PHONY: clean
