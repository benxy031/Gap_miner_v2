# Makefile for GapMiner V2
# Phase 1-5: Core modules + Real RPC integration

CC = gcc
CFLAGS = -Wall -Wextra -O2 -g -std=c99
CFLAGS += $(shell pkg-config --cflags libcurl jansson)
LDFLAGS = -lm -lgmp -lssl -lcrypto
LDFLAGS += $(shell pkg-config --libs libcurl jansson)

# Directories
SRC_DIR = new_src
TEST_DIR = tests
BUILD_DIR = build
BIN_DIR = bin

# Optional CUDA GPU Fermat acceleration (new_src/gpu/gpu_fermat.cu).
# Enable with: make WITH_CUDA=1 [WITH_CGBN_FERMAT=1] [GPU_BITS=1280] [CUDA_ARCH=-arch=sm_86]
# GPU_BITS default (1280 = 20 limbs) covers the full non-CRT shift range up to
# 1024 (h256 256 bits + shift 1024 = 1280 bits). The active limb count is
# narrowed at runtime per shift (gpu_adapter_set_candidate_bits), so low
# shifts still run at their actual width (e.g. AL=6 at shift 26) and land on
# CGBN's AL=6/TPI=4 fast path when WITH_CGBN_FERMAT=1. Without WITH_CUDA,
# gpu_adapter.c falls back to the existing CPU (GMP) Fermat test unchanged.
GPU_BITS ?= 1280
GPU_NLIMBS := $(shell echo '$(GPU_BITS) / 64' | bc)

ifdef WITH_CUDA
NVCC ?= nvcc
CUDA_ARCH ?= -arch=sm_86
# Auto-detect the CUDA toolkit: prefer /usr/local/cuda, else the distro
# system install (headers in /usr/include, libcudart in /usr/lib/x86_64-linux-gnu).
CUDA_PATH ?= $(shell test -d /usr/local/cuda && echo /usr/local/cuda || echo /usr)
CUDA_LIBDIR ?= $(shell test -d /usr/local/cuda/lib64 && echo /usr/local/cuda/lib64 || echo /usr/lib/x86_64-linux-gnu)
CFLAGS += -DWITH_CUDA -DGPU_NLIMBS=$(GPU_NLIMBS) -I$(CUDA_PATH)/include
NVCC_FLAGS = -DGPU_NLIMBS=$(GPU_NLIMBS) -std=c++17
LDFLAGS += -lcudart -L$(CUDA_LIBDIR)

# Optional CGBN-based Fermat kernel (~1.9x faster for even active-limb widths).
# Requires internet on first build (auto-clones NVlabs/CGBN header-only library).
CGBN_DIR := tools/cgbn
CGBN_HDR := $(CGBN_DIR)/include/cgbn/cgbn.h
ifdef WITH_CGBN_FERMAT
NVCC_FLAGS += -DWITH_CGBN_FERMAT -I$(CGBN_DIR)/include
CFLAGS += -DWITH_CGBN_FERMAT
endif

GPU_OBJ = $(BUILD_DIR)/$(SRC_DIR)/gpu/gpu_fermat.o \
		  $(BUILD_DIR)/$(SRC_DIR)/gpu/gpu_sieve.o
else
GPU_OBJ =
endif

# Source files
SOURCES = $(SRC_DIR)/sieve_core.c \
          $(SRC_DIR)/gap_detection.c \
          $(SRC_DIR)/gap_dist.c \
          $(SRC_DIR)/halfclass.c \
          $(SRC_DIR)/primality_fermat.c \
          $(SRC_DIR)/primality_euler.c \
          $(SRC_DIR)/primality_bpsw.c \
          $(SRC_DIR)/primality_limbs.c \
          $(SRC_DIR)/atomic_nonce.c \
          $(SRC_DIR)/gap_priority.c \
          $(SRC_DIR)/crt_set.c \
          $(SRC_DIR)/gap_target.c \
          $(SRC_DIR)/covering.c \
          $(SRC_DIR)/crt_runtime.c \
          $(SRC_DIR)/worker_gpu.c \
          $(SRC_DIR)/miner_farm.c \
          $(SRC_DIR)/gpu_adapter.c \
          $(SRC_DIR)/gapcoin_rpc.c \
		  $(SRC_DIR)/gapcoin_work.c \
          $(SRC_DIR)/block_assembly.c \
          $(SRC_DIR)/submission_pipeline.c \
          $(SRC_DIR)/merit_records.c \
          $(SRC_DIR)/record_log.c

OBJECTS = $(SOURCES:%.c=$(BUILD_DIR)/%.o) $(GPU_OBJ)

# Main binary and tests
MAIN_BINARY = $(BIN_DIR)/gapminer
TEST_GAP_DETECTION = $(BIN_DIR)/test_gap_detection
TEST_PRIMALITY = $(BIN_DIR)/test_primality
TEST_WORKER_THREADS = $(BIN_DIR)/test_worker_threads
TEST_GAPCOIN_RPC = $(BIN_DIR)/test_gapcoin_rpc
TEST_BLOCK_SUBMISSION = $(BIN_DIR)/test_block_submission
TEST_SIEVE_CORE = $(BIN_DIR)/test_sieve_core
TEST_GPU_FERMAT = $(BIN_DIR)/test_gpu_fermat
TEST_GPU_SIEVE = $(BIN_DIR)/test_gpu_sieve
TEST_GAP_DIST = $(BIN_DIR)/test_gap_dist
TEST_HALFCLASS = $(BIN_DIR)/test_halfclass
TEST_GAP_PRIORITY = $(BIN_DIR)/test_gap_priority
TEST_CRT_SET = $(BIN_DIR)/test_crt_set
CRT_GEN = $(BIN_DIR)/crt_gen
TEST_GAP_TARGET = $(BIN_DIR)/test_gap_target
TEST_COVERING = $(BIN_DIR)/test_covering
GEN_CRT = $(BIN_DIR)/gen_crt
TEST_CRT_RUNTIME = $(BIN_DIR)/test_crt_runtime
TEST_CRT_SUBMISSION = $(BIN_DIR)/test_crt_submission
BENCH_FERMAT = $(BIN_DIR)/bench_fermat

# Phony targets
.PHONY: all clean test help update-merits

# Default target
all: $(BIN_DIR) $(BUILD_DIR) $(OBJECTS) $(MAIN_BINARY) $(TEST_GAP_DETECTION) $(TEST_PRIMALITY) $(TEST_WORKER_THREADS) $(TEST_GAPCOIN_RPC) $(TEST_BLOCK_SUBMISSION) $(TEST_SIEVE_CORE) $(TEST_GPU_FERMAT) $(TEST_GPU_SIEVE) $(TEST_GAP_DIST) $(TEST_HALFCLASS) $(TEST_GAP_PRIORITY) $(TEST_CRT_SET) $(TEST_GAP_TARGET) $(TEST_COVERING) $(TEST_CRT_RUNTIME) $(TEST_CRT_SUBMISSION) $(CRT_GEN) $(GEN_CRT)

# Create directories
$(BIN_DIR):
	@mkdir -p $(BIN_DIR)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BUILD_DIR)/$(SRC_DIR)
	@mkdir -p $(BUILD_DIR)/$(TEST_DIR)

# Compile source files
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Link main binary
$(MAIN_BINARY): $(OBJECTS) $(BUILD_DIR)/$(SRC_DIR)/main.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -lpthread -o $@
	@echo "✓ Built: $@"

# Link test_gap_detection
$(TEST_GAP_DETECTION): $(OBJECTS) $(BUILD_DIR)/$(TEST_DIR)/test_gap_detection.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@
	@echo "✓ Built: $@"

# Link test_primality
$(TEST_PRIMALITY): $(OBJECTS) $(BUILD_DIR)/$(TEST_DIR)/test_primality.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@
	@echo "✓ Built: $@"

# Link test_worker_threads
$(TEST_WORKER_THREADS): $(OBJECTS) $(BUILD_DIR)/$(TEST_DIR)/test_worker_threads.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -lpthread -o $@
	@echo "✓ Built: $@"

# Link test_gapcoin_rpc
$(TEST_GAPCOIN_RPC): $(OBJECTS) $(BUILD_DIR)/$(TEST_DIR)/test_gapcoin_rpc.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -lpthread -o $@
	@echo "✓ Built: $@"

# Link test_block_submission
$(TEST_BLOCK_SUBMISSION): $(OBJECTS) $(BUILD_DIR)/$(TEST_DIR)/test_block_submission.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -lpthread -o $@
	@echo "✓ Built: $@"

$(TEST_SIEVE_CORE): $(OBJECTS) $(BUILD_DIR)/$(TEST_DIR)/test_sieve_core.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -lpthread -o $@
	@echo "✓ Built: $@"

# Link test_gpu_fermat (self-skips at runtime when built without WITH_CUDA=1)
$(TEST_GPU_FERMAT): $(OBJECTS) $(BUILD_DIR)/$(TEST_DIR)/test_gpu_fermat.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -lpthread -o $@
	@echo "✓ Built: $@"

# Link test_gpu_sieve (self-skips at runtime when built without WITH_CUDA=1)
$(TEST_GPU_SIEVE): $(OBJECTS) $(BUILD_DIR)/$(TEST_DIR)/test_gpu_sieve.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -lpthread -o $@
	@echo "✓ Built: $@"

# Link test_gap_dist
$(TEST_GAP_DIST): $(BUILD_DIR)/$(SRC_DIR)/gap_dist.o $(BUILD_DIR)/$(TEST_DIR)/test_gap_dist.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@
	@echo "✓ Built: $@"

# Link test_halfclass (HALF_CLASS two-pass parity vs the full-class pipeline)
$(TEST_HALFCLASS): $(BUILD_DIR)/$(SRC_DIR)/halfclass.o $(BUILD_DIR)/$(SRC_DIR)/gap_detection.o $(BUILD_DIR)/$(SRC_DIR)/gap_dist.o $(BUILD_DIR)/$(SRC_DIR)/sieve_core.o $(BUILD_DIR)/$(TEST_DIR)/test_halfclass.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -lpthread -o $@
	@echo "✓ Built: $@"

# GPU Fermat kernel throughput benchmark (dev tool)
$(BENCH_FERMAT): $(OBJECTS) $(BUILD_DIR)/tools/bench_fermat.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -lpthread -o $@
	@echo "✓ Built: $@"

# Link test_gap_priority
$(TEST_GAP_PRIORITY): $(OBJECTS) $(BUILD_DIR)/$(TEST_DIR)/test_gap_priority.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -lpthread -o $@
	@echo "✓ Built: $@"

# CRT set module (lightweight: only crt_set.o + its consumer, no CUDA needed)
$(TEST_CRT_SET): $(BUILD_DIR)/$(SRC_DIR)/crt_set.o $(BUILD_DIR)/$(TEST_DIR)/test_crt_set.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -lpthread -o $@
	@echo "✓ Built: $@"

$(CRT_GEN): $(BUILD_DIR)/$(SRC_DIR)/crt_set.o $(BUILD_DIR)/$(SRC_DIR)/crt_gen_main.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -lpthread -o $@
	@echo "✓ Built: $@"

$(GEN_CRT): $(BUILD_DIR)/$(SRC_DIR)/covering.o $(BUILD_DIR)/$(SRC_DIR)/gen_crt_main.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -lm -o $@
	@echo "✓ Built: $@"

$(TEST_CRT_RUNTIME): $(BUILD_DIR)/$(SRC_DIR)/covering.o $(BUILD_DIR)/$(SRC_DIR)/crt_runtime.o $(BUILD_DIR)/$(TEST_DIR)/test_crt_runtime.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -lm -lgmp -o $@
	@echo "✓ Built: $@"

# CRT big-nAdd submission assembly (links the full object set like the other
# integration tests; CPU-only in `make test`).
$(TEST_CRT_SUBMISSION): $(OBJECTS) $(BUILD_DIR)/$(TEST_DIR)/test_crt_submission.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -lpthread -o $@
	@echo "✓ Built: $@"

$(TEST_GAP_TARGET): $(BUILD_DIR)/$(SRC_DIR)/gap_target.o $(BUILD_DIR)/$(TEST_DIR)/test_gap_target.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -lpthread -o $@
	@echo "✓ Built: $@"

$(TEST_COVERING): $(BUILD_DIR)/$(SRC_DIR)/covering.o $(BUILD_DIR)/$(TEST_DIR)/test_covering.o | $(BIN_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -lm -o $@
	@echo "✓ Built: $@"

# Compile test objects
$(BUILD_DIR)/$(TEST_DIR)/%.o: $(TEST_DIR)/%.c
	@mkdir -p $(BUILD_DIR)/$(TEST_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile dev tools
$(BUILD_DIR)/tools/%.o: tools/%.c
	@mkdir -p $(BUILD_DIR)/tools
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA GPU sources (only reached when WITH_CUDA=1 selects a .o under gpu/)
$(BUILD_DIR)/$(SRC_DIR)/gpu/%.o: $(SRC_DIR)/gpu/%.cu | $(BUILD_DIR)
	@mkdir -p $(dir $@)
ifdef WITH_CGBN_FERMAT
	@if [ ! -f "$(CGBN_HDR)" ]; then \
		echo "[cgbn] Cloning NVlabs/CGBN (header-only)..."; \
		git clone --depth=1 https://github.com/NVlabs/CGBN.git $(CGBN_DIR); \
	fi
endif
	$(NVCC) -O3 $(CUDA_ARCH) $(NVCC_FLAGS) -c $< -o $@

# Run tests
test: $(TEST_GAP_DETECTION) $(TEST_PRIMALITY) $(TEST_WORKER_THREADS) $(TEST_GAPCOIN_RPC) $(TEST_BLOCK_SUBMISSION) $(TEST_SIEVE_CORE) $(TEST_GPU_FERMAT) $(TEST_GPU_SIEVE) $(TEST_GAP_DIST) $(TEST_HALFCLASS) $(TEST_GAP_PRIORITY) $(TEST_CRT_SET) $(TEST_GAP_TARGET) $(TEST_COVERING) $(TEST_CRT_RUNTIME) $(TEST_CRT_SUBMISSION)
	@echo "\n========== Running Integration Tests =========="
	@$(TEST_GAP_DETECTION)
	@echo ""
	@$(TEST_PRIMALITY)
	@echo ""
	@$(TEST_WORKER_THREADS)
	@echo ""
	@$(TEST_GAPCOIN_RPC)
	@echo ""
	@$(TEST_BLOCK_SUBMISSION)
	@echo ""
	@$(TEST_SIEVE_CORE)
	@echo ""
	@$(TEST_GPU_FERMAT)
	@echo ""
	@$(TEST_GPU_SIEVE)
	@echo ""
	@$(TEST_GAP_DIST)
	@echo ""
	@$(TEST_HALFCLASS)
	@echo ""
	@$(TEST_GAP_PRIORITY)
	@echo ""
	@$(TEST_CRT_SET)
	@echo ""
	@$(TEST_GAP_TARGET)
	@echo ""
	@$(TEST_COVERING)
	@echo ""
	@$(TEST_CRT_RUNTIME)
	@echo ""
	@$(TEST_CRT_SUBMISSION)

# Clean build artifacts
clean:
	@rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "✓ Cleaned build artifacts"

# Refresh data/prime_gap_merits.txt from primegaps.cloudygo.com (requires internet)
update-merits:
	@bash scripts/update_merits.sh

# Help
help:
	@echo "GapMiner V2 — Phase 1 Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all             - Build all object files and test binaries"
	@echo "  test            - Run integration tests"
	@echo "  clean           - Remove build artifacts"
	@echo "  update-merits   - Refresh data/prime_gap_merits.txt from primegaps.cloudygo.com"
	@echo "  help            - Show this help message"
	@echo ""
	@echo "Example: make test"
