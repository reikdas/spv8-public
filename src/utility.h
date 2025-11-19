#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define always_inline __inline__ __attribute__((always_inline))
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

typedef unsigned int u32;
typedef unsigned long long u64;

struct csr_matrix {
  int m, rows, cols;
  double *nnz, *x, *y, *ans;
  int *col, *rowb, *rowe;
  int *tstart;
  int *tend;
};

void input_matrix(struct csr_matrix *mat);
void destroy_matrix(struct csr_matrix *mat);
int check_answer(struct csr_matrix *mat);
struct csr_matrix apply_order(struct csr_matrix *mat, int **tasks, int *task_sizes, int task_count, int copy_oob);
