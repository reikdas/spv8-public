# Compilers
CC = gcc

CFLAGS += -O3 -Wall -Wextra -mavx512f -mavx512vl -mfma -mprfchw

LDFLAGS  += -lm

VPATH = src

OBJ = bin/spmv_spv8.o
MAIN_OBJ = bin/spmv_spv8_main.o
TEST_OBJ = bin/spmv_spv8_test.o

all: bin bin/spmv_spv8

bin:
	mkdir -p bin

bin/spmv_spv8: $(OBJ) $(MAIN_OBJ)
	$(CC) $(CFLAGS) $(OBJ) $(MAIN_OBJ) -o $@ $(LDFLAGS)

bin/spmv_spv8_test: $(OBJ) $(TEST_OBJ)
	$(CC) $(CFLAGS) $(OBJ) $(TEST_OBJ) -o $@ $(LDFLAGS)

.PHONY : test
test: bin/spmv_spv8_test
	./bin/spmv_spv8_test

bin/%.o: src/%.c | bin
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY : clean
clean :
	-rm bin/*

