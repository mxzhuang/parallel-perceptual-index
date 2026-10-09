CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
# no -ffast-math: the port must keep IEEE semantics to match the reference
SRC  = src/main.cpp src/ma.cpp src/niqe.cpp src/ggd.cpp src/matlab_compat.cpp src/random_forest.cpp
OBJ  = $(SRC:.cpp=.o)
BIN  = pi_eval

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ)

src/main.o: CXXFLAGS += -Wno-unused-function -Wno-missing-field-initializers

%.o: %.cpp src/*.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(BIN)
	./$(BIN) --timing tests/images/*_y.png

# accuracy checks: GGD lookup equivalence + comparison with the official reference
check: $(BIN) tests/test_ggd
	./tests/test_ggd
	./$(BIN) --dump tests/cpp_dump.txt tests/images/*_y.png > /dev/null
	python3 tests/compare.py reference/octave_reference.txt tests/cpp_dump.txt

tests/test_ggd: tests/test_ggd.cpp src/ggd.cpp src/matlab_compat.cpp src/*.h
	$(CXX) $(CXXFLAGS) tests/test_ggd.cpp src/ggd.cpp src/matlab_compat.cpp -o $@

clean:
	rm -f $(OBJ) $(BIN) tests/test_ggd tests/cpp_dump.txt

.PHONY: all test check clean
