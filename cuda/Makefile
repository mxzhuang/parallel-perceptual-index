# Optional CUDA extension: builds pi_eval_cuda from the CPU sources in ../src (unchanged)
# plus the kernels in this directory. Needs the CUDA toolkit (nvcc) and an NVIDIA GPU.
#
#   make -C cuda            build ../pi_eval_cuda
#   make -C cuda check      run it on the test images and compare with the CPU modules
NVCC     ?= nvcc
CXX      ?= g++
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wextra -ffp-contract=off
# --fmad=false: no fused multiply-add, so per-pixel values round as on the CPU build
NVCCFLAGS ?= -O2 -std=c++17 --fmad=false

SRC = $(filter-out ../src/main.cpp, $(wildcard ../src/*.cpp))
OBJ = $(SRC:../src/%.cpp=build/%.o)
HDR = $(wildcard ../src/*.h) gpu_kernels.h
BIN = ../pi_eval_cuda

all: $(BIN)

$(BIN): $(OBJ) build/main_cuda.o build/gpu_kernels.o
	$(NVCC) -Xcompiler -fopenmp -o $@ $^ -lgomp

build/%.o: ../src/%.cpp $(HDR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -fopenmp -c $< -o $@

build/image_io.o: CXXFLAGS += -Wno-unused-function -Wno-missing-field-initializers

build/main_cuda.o: main_cuda.cpp $(HDR)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -fopenmp -c $< -o $@

build/gpu_kernels.o: gpu_kernels.cu gpu_kernels.h
	@mkdir -p build
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

check: $(BIN)
	cd .. && ./pi_eval_cuda tests/images/*_y.png

clean:
	rm -rf build $(BIN)

.PHONY: all check clean
