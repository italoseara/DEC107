##### Configurable variables (override on make command line)
# Algorithm: serial | openmp | mpi
ALG ?= serial
# Threads for OpenMP; for mpi, use RANKS instead
THREADS ?= 1
# MPI ranks for running mpi algorithm
RANKS ?= 2
# Matrix sizes
M ?= 512
N ?= 512
K ?= 512

##### Compiler and flags
MPICC   ?= mpicc
CFLAGS  ?= -O2 -Wall -Wextra -Wunused-parameter -std=c11 -fopenmp
LDFLAGS ?= -fopenmp
LDLIBS  ?= -lm
# BLAS/CBLAS linkage for tests (override if needed, e.g., BLAS_LIBS="-lcblas -lblas")
BLAS_LIBS ?= -lopenblas

##### Paths (relative to this Makefile location)
SRC_DIR   := src
INC_DIR   := include
BUILD_DIR := build

##### Files
SRCS := $(SRC_DIR)/dgemm.c $(SRC_DIR)/main.c
BIN  := $(BUILD_DIR)/dgemm_run

# Tests
TEST_SRCS := $(SRC_DIR)/dgemm.c tests/test_dgemm.c
TEST_BIN  := $(BUILD_DIR)/test_dgemm

.PHONY: all run clean distclean help

all: $(BIN)

$(BUILD_DIR):
	mkdir -p "$@"

$(BIN): $(SRCS) | $(BUILD_DIR)
	$(MPICC) $(CFLAGS) -I"$(INC_DIR)" $(SRCS) -o "$@" $(LDFLAGS) $(LDLIBS)

# Build the unit test binary for DGEMM (links with BLAS for cblas_dgemm)
$(TEST_BIN): $(TEST_SRCS) | $(BUILD_DIR)
	$(MPICC) $(CFLAGS) -I"$(INC_DIR)" $(TEST_SRCS) -o "$@" $(LDFLAGS) $(LDLIBS) $(BLAS_LIBS)

# Example: make run ALG=openmp THREADS=4 M=1024 N=1024 K=1024
run: $(BIN)
	@if [ "$(ALG)" = "mpi" ]; then \
	  mpirun -np $(RANKS) "$(BIN)" --alg mpi --m $(M) --n $(N) --k $(K); \
	else \
	  "$(BIN)" --alg $(ALG) --threads $(THREADS) --m $(M) --n $(N) --k $(K); \
	fi

help:
	@echo "Targets:"
	@echo "  all            Build the runner"
	@echo "  run            Build and run with configurable vars"
	@echo "  test           Build and run tests/test_dgemm"
	@echo "  clean          Remove build artifacts"
	@echo "Variables (override on command line):"
	@echo "  ALG=serial|openmp|mpi   (default: serial)"
	@echo "  THREADS=<int>           (default: 1; used by openmp)"
	@echo "  RANKS=<int>             (default: 2; used by mpi)"
	@echo "  M=<int> N=<int> K=<int> (default: 512 512 512)"
	@echo "  BLAS_LIBS=...           (default: -lopenblas; e.g., \"-lcblas -lblas\")"

clean:
	$(RM) -r "$(BUILD_DIR)"

distclean: clean

# Build and run unit tests
.PHONY: test
test: $(TEST_BIN)
	"$(TEST_BIN)"
