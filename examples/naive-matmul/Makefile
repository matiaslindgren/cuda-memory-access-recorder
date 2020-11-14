NVCC=nvcc
NVCCFLAGS=--std=c++17 -O3 -I../../include
MAIN_SRC=main.cu
OUTPUT_DIR=bin

.PHONY: clean all dirs main

all: dirs main
dirs:
	@mkdir -pv $(OUTPUT_DIR)
main:
	@$(NVCC) $(NVCCFLAGS) $(MAIN_SRC) --output-file $(OUTPUT_DIR)/main
clean:
	@rm -rfv $(OUTPUT_DIR) __pycache__ matrix.txt result.txt
