CXX      ?= g++
# -g only adds debug symbols (for perf); it does not change the generated code.
# No -ffast-math: the port must keep IEEE semantics to match the reference.
# -ffp-contract=off: never fuse a multiply and an add, so every build rounds the same way
# (plain x86-64 has no FMA, so this changes nothing there; it matters with -march or on ARM).
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wextra -ffp-contract=off
OMPFLAGS ?= -fopenmp

SRC = $(wildcard src/*.cpp)
HDR = $(wildcard src/*.h)

# pi_eval      all versions (sequential baseline and the OpenMP versions), built with OpenMP
# pi_eval_seq  the same sources without OpenMP: every pragma is ignored, so this binary is
#              the pure sequential baseline that speedups are measured against
OMP_OBJ = $(SRC:src/%.cpp=build/omp/%.o)
SEQ_OBJ = $(SRC:src/%.cpp=build/seq/%.o)

all: pi_eval pi_eval_seq

pi_eval: $(OMP_OBJ)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $(OMP_OBJ)

pi_eval_seq: $(SEQ_OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(SEQ_OBJ)

build/omp/%.o: src/%.cpp $(HDR)
	@mkdir -p build/omp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c $< -o $@

build/seq/%.o: src/%.cpp $(HDR)
	@mkdir -p build/seq
	$(CXX) $(CXXFLAGS) -Wno-unknown-pragmas -c $< -o $@

build/omp/image_io.o build/seq/image_io.o build/gprof/image_io.o: CXXFLAGS += -Wno-unused-function -Wno-missing-field-initializers

# gprof build of the sequential baseline (writes gmon.out when it exits)
GPROF_OBJ = $(SRC:src/%.cpp=build/gprof/%.o)

gprof: pi_eval_gprof

pi_eval_gprof: $(GPROF_OBJ)
	$(CXX) $(CXXFLAGS) -pg -o $@ $(GPROF_OBJ)

build/gprof/%.o: src/%.cpp $(HDR)
	@mkdir -p build/gprof
	$(CXX) $(CXXFLAGS) -Wno-unknown-pragmas -pg -c $< -o $@

test: pi_eval
	./pi_eval --timing tests/images/*_y.png

# Accuracy checks:
#   1. GGD lookup: binary search returns the same index as the linear search
#   2. sequential baseline against the reference outputs of the official code
#   3. every parallel configuration against the sequential baseline
check: pi_eval pi_eval_seq tests/test_ggd
	./tests/test_ggd
	./pi_eval_seq --dump tests/cpp_dump.txt tests/images/*_y.png > /dev/null
	python3 tests/compare.py reference/octave_reference.txt tests/cpp_dump.txt
	python3 tests/check_parallel.py

tests/test_ggd: tests/test_ggd.cpp src/ggd.cpp src/matlab_compat.cpp $(HDR)
	$(CXX) $(CXXFLAGS) tests/test_ggd.cpp src/ggd.cpp src/matlab_compat.cpp -o $@

# profiling with time, gprof and perf (see tools/profile.sh)
profile: pi_eval_seq pi_eval_gprof
	bash tools/profile.sh tests/images/*_y.png

clean:
	rm -rf build pi_eval pi_eval_seq pi_eval_gprof tests/test_ggd tests/cpp_dump.txt tests/out

.PHONY: all gprof test check profile clean
