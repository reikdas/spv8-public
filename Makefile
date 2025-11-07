# Compilers
CC = gcc
CXX = g++
# Flags
# - Debug
# CXXFLAGS += -g
# CXXFLAGS += -fsanitize=address # Check invalid memory access
# - General
CXXFLAGS += -O3 # Should be disabled when debug
CXXFLAGS += -std=c++17 -Wall -Wextra
CXXFLAGS += -march=native -fopenmp
#CXXFLAGS += -static

# Intel oneMKL setup (will try to infer MKLROOT if not set)
MKLROOT ?= $(HOME)/intel/oneapi/mkl/latest
ifeq (,$(wildcard $(MKLROOT)))
$(warning MKLROOT is not set or invalid. Consider sourcing oneAPI: `source ~/intel/oneapi/setvars.sh`)
endif

MKL_INC ?= $(MKLROOT)/include
MKL_LIB ?= $(MKLROOT)/lib/intel64
COMPILER_LIB ?= $(HOME)/intel/oneapi/compiler/latest/lib

# Include and link flags for MKL (Intel OpenMP threading)
CXXFLAGS += -I$(MKL_INC)
LDFLAGS  += -L$(MKL_LIB) -L$(COMPILER_LIB)
LDFLAGS  += -Wl,--start-group -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -Wl,--end-group
LDFLAGS  += -liomp5

# To run without explicitly setting LD_LIBRARY_PATH
LDFLAGS  += -Wl,-rpath,$(MKL_LIB) -Wl,-rpath,$(COMPILER_LIB)

VPATH = src

all: bin bin/spmv_spv8 bin/spmv_mkl

bin:
	mkdir -p bin

bin/spmv_spv8 : spmv_spv8.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

bin/spmv_mkl : spmv_mkl.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

.PHONY : clean
clean :
	-rm bin/*

