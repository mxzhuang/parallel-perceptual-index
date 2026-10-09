CXX      ?= g++
# -g only adds debug symbols (for perf); it does not change the generated code.
# No -ffast-math: the port must keep IEEE semantics to match the reference.
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wextra
SRC  = src/main.cpp src/ma.cpp src/niqe.cpp src/ggd.cpp src/matlab_compat.cpp src/random_forest.cpp
OBJ  = $(SRC:.cpp=.o)
BIN  = pi_eval

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ)

src/main.o build/gprof/main.o: CXXFLAGS += -Wno-unused-function -Wno-missing-field-initializers

%.o: %.cpp src/*.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# gprof build: same code instrumented with -pg (writes gmon.out when it exits)
GPROF_OBJ = $(SRC:src/%.cpp=build/gprof/%.o)

gprof: pi_eval_gprof

pi_eval_gprof: $(GPROF_OBJ)
	$(CXX) $(CXXFLAGS) -pg -o $@ $(GPROF_OBJ)

build/gprof/%.o: src/%.cpp src/*.h
	@mkdir -p build/gprof
	$(CXX) $(CXXFLAGS) -pg -c $< -o $@

test: $(BIN)
	./$(BIN) --timing tests/images/*_y.png

# accuracy checks: GGD lookup equivalence + comparison with the official reference
check: $(BIN) tests/test_ggd
	./tests/test_ggd
	./$(BIN) --dump tests/cpp_dump.txt tests/images/*_y.png > /dev/null
	python3 tests/compare.py reference/octave_reference.txt tests/cpp_dump.txt

tests/test_ggd: tests/test_ggd.cpp src/ggd.cpp src/matlab_compat.cpp src/*.h
	$(CXX) $(CXXFLAGS) tests/test_ggd.cpp src/ggd.cpp src/matlab_compat.cpp -o $@

# profiling with time, gprof and perf (see tools/profile.sh)
profile: $(BIN) pi_eval_gprof
	bash tools/profile.sh tests/images/*_y.png

clean:
	rm -rf $(OBJ) $(BIN) pi_eval_gprof build tests/test_ggd tests/cpp_dump.txt

.PHONY: all gprof test check profile clean
