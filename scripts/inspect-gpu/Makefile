NVCC=/opt/cuda/bin/nvcc
NVCCFLAGS=--std=c++17 -O3

.PHONY: all dirs analyze clean

all: dirs analyze

dirs:
	@mkdir -p bin

analyze:
	@$(NVCC) $(NVCCFLAGS) analyze.cu -o bin/analyze

clean:
	@rm -rfv bin
