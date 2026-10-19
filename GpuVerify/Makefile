# GpuVerify - standalone GPU prime-gap verifier.
# Depends only on CUDA (nvcc, sm_86 for the RTX 3070) and GMP.
# Independent of the gapminer and phase0 builds.

NVCC   ?= nvcc
CC     ?= gcc
ARCH   ?= sm_86
OBJ    := obj
BIN    := bin/gpuverify

CFLAGS := -O2 -Wall -Wextra -Isrc
NVFLAGS := -O3 -arch=$(ARCH) -Xcompiler=-O2 -Xcompiler=-Wall -Isrc
LDLIBS := -lgmp -lm

SRC_C := src/main.c src/parse.c
SRC_CU := src/host.cu src/prp_kernel.cu src/sieve_kernel.cu
OBJS := $(OBJ)/main.o $(OBJ)/parse.o $(OBJ)/host.o $(OBJ)/prp_kernel.o \
        $(OBJ)/sieve_kernel.o

.PHONY: all clean test test-parser

all: $(BIN)

$(BIN): $(OBJS)
	$(NVCC) -o $@ $(OBJS) $(LDLIBS)

$(OBJ)/main.o: src/main.c src/gv.h src/host.h src/parse.h | $(OBJ)
	$(CC) $(CFLAGS) -c -o $@ src/main.c

$(OBJ)/parse.o: src/parse.c src/parse.h src/gv.h | $(OBJ)
	$(CC) $(CFLAGS) -c -o $@ src/parse.c

$(OBJ)/host.o: src/host.cu src/host.h src/gv.h | $(OBJ)
	$(NVCC) $(NVFLAGS) -c -o $@ src/host.cu

$(OBJ)/prp_kernel.o: src/prp_kernel.cu src/mont.cuh src/gv.h | $(OBJ)
	$(NVCC) $(NVFLAGS) -c -o $@ src/prp_kernel.cu

$(OBJ)/sieve_kernel.o: src/sieve_kernel.cu | $(OBJ)
	$(NVCC) $(NVFLAGS) -c -o $@ src/sieve_kernel.cu

$(OBJ):
	mkdir -p $(OBJ) bin

test-parser: $(OBJ)/test_parse
	./$(OBJ)/test_parse /tmp/allgaps.sql

$(OBJ)/test_parse: tests/test_parse.c src/parse.c src/parse.h src/gv.h | $(OBJ)
	$(CC) $(CFLAGS) -o $@ tests/test_parse.c src/parse.c $(LDLIBS)

test-mont: bin/test_mont
	./bin/test_mont

bin/test_mont: tests/test_mont.cu src/mont.cuh src/gv.h | $(OBJ)
	$(NVCC) $(NVFLAGS) -o $@ tests/test_mont.cu $(LDLIBS)

test: $(BIN)
	./$(BIN) selftest

clean:
	rm -rf $(OBJ) bin
