NVCC=nvcc
NVCCFLAGS=--std=c++14 -O3 -I../include
MAIN_SRC=main.cu
OUTPUT_DIR=bin

.PHONY: clean all

all: dirs main
dirs:
	mkdir --parents $(OUTPUT_DIR)
main:
	$(NVCC) $(NVCCFLAGS) $(MAIN_SRC) --output-file $(OUTPUT_DIR)/main
clean:
	@if [ ! -d "$(OUTPUT_DIR)" ]; then echo already clean; else rm --recursive --verbose $(OUTPUT_DIR); fi
