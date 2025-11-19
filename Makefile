# Compilers
CC = gcc
CXX = g++
CC = gcc
# Flags
# - Debug
# CXXFLAGS += -g
# CXXFLAGS += -fsanitize=address # Check invalid memory access
# - General
CXXFLAGS += -O3 # Should be disabled when debug
CXXFLAGS += -std=c++17 -Wall -Wextra
CXXFLAGS += -march=native
CFLAGS += -O3 -Wall -Wextra -march=native
#CXXFLAGS += -static

LDFLAGS  += -lm

VPATH = src

all: bin bin/spmv_spv8

bin:
	mkdir -p bin

bin/spmv_spv8 : spmv_spv8.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)



.PHONY : clean
clean :
	-rm bin/*

