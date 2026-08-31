#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "utility.h"

/* Correctness test: compares spmv_tr_spvv8_kernel against a plain CSR
 * SpMV reference on matrices chosen to exercise every kernel path:
 *   - the scalar avx512_fma_spvv kernel (remainder rows, rowlen > 32)
 *   - the bulk avx512_spvv8 kernel (groups of 8 same-length rows)
 *   - empty rows, banded matrices, multiple panels
 */

/* Ported from upstream utility.hpp check_answer() */
static int check_answer(int rows, const double *y, const double *ans) {
  int bad_count = 0;
  for (int i = 0; i < rows; i++) {
    double yi = y[i];
    double ansi = ans[i];
    if (fabs(yi - ansi) > 0.01 * fabs(ansi) && !(fabs(yi) <= 1e-5 && fabs(ansi) <= 1e-5)) {
      if (bad_count < 10)
        fprintf(stderr, "  y[%d] expected %lf got %lf\n", i, ansi, yi);
      bad_count++;
    }
  }
  if (bad_count)
    fprintf(stderr, "  bad_count: %d\n", bad_count);
  return bad_count == 0;
}

static void reference_spmv(int rows, const double *nnz, const int *col,
                           const int *indptr, const double *x, double *y) {
  for (int r = 0; r < rows; r++) {
    double sum = 0.0;
    for (int i = indptr[r]; i < indptr[r + 1]; i++)
      sum += nnz[i] * x[col[i]];
    y[r] = sum;
  }
}

static double rand_val(void) {
  return 2.0 * ((double)rand() / RAND_MAX) - 1.0;
}

/* Pick `rowlen` distinct sorted columns out of [0, cols) */
static int cmp_int(const void *a, const void *b) {
  return *(const int *)a - *(const int *)b;
}

static void pick_columns(int cols, int rowlen, int *out) {
  static int *cand = NULL;
  static int cand_cap = 0;
  if (cols > cand_cap) {
    cand = (int *)realloc(cand, cols * sizeof(int));
    cand_cap = cols;
  }
  for (int i = 0; i < cols; i++) cand[i] = i;
  for (int i = 0; i < rowlen; i++) {
    int j = i + rand() % (cols - i);
    int tmp = cand[i]; cand[i] = cand[j]; cand[j] = tmp;
  }
  memcpy(out, cand, rowlen * sizeof(int));
  qsort(out, rowlen, sizeof(int), cmp_int);
}

/* Run one test case on a CSR matrix; returns 1 on pass */
static int run_case(const char *name, int rows, int cols, int m,
                    double *nnz, int *col, int *indptr) {
  struct csr_matrix mat = input_matrix(m, rows, cols, nnz, col, indptr);

  double *x = (double *)malloc(cols * sizeof(double));
  double *y = (double *)malloc(rows * sizeof(double));
  double *ans = (double *)malloc(rows * sizeof(double));
  for (int i = 0; i < cols; i++) x[i] = rand_val();

  reference_spmv(rows, nnz, col, indptr, x, ans);

  struct tr_matrix tr = process(&mat);
  for (int i = 0; i < rows; i++) y[i] = 0.0;
  spmv_tr_spvv8_kernel(&tr, x, y);

  int ok = check_answer(rows, y, ans);
  printf("%-40s %s\n", name, ok ? "PASS" : "FAIL");

  destroy_matrix(&mat);
  for (int t = 0; t < tr.task_count; t++) free(tr.tasks[t]);
  free(tr.tasks); free(tr.task_sizes); free(tr.spvv8_len);
  destroy_matrix(&tr.mat);
  free(x); free(y); free(ans);

  return ok;
}

/* Build a random CSR matrix where row r gets length rowlens[r] */
static int run_random_case(const char *name, int rows, int cols,
                           const int *rowlens) {
  int m = 0;
  for (int r = 0; r < rows; r++) m += rowlens[r];

  double *nnz = (double *)malloc((m ? m : 1) * sizeof(double));
  int *col = (int *)malloc((m ? m : 1) * sizeof(int));
  int *indptr = (int *)malloc((rows + 1) * sizeof(int));

  indptr[0] = 0;
  for (int r = 0; r < rows; r++) {
    int rl = rowlens[r];
    pick_columns(cols, rl, col + indptr[r]);
    for (int i = 0; i < rl; i++) nnz[indptr[r] + i] = rand_val();
    indptr[r + 1] = indptr[r] + rl;
  }

  int ok = run_case(name, rows, cols, m, nnz, col, indptr);
  free(nnz); free(col); free(indptr);
  return ok;
}

int main(void) {
  srand(12345);
  int failures = 0;
  int *rowlens;

  /* 1. Tiny 4x5 (same shape as the demo in spmv_spv8_main.c):
     everything goes through the scalar remainder path */
  {
    double nnz[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int col[6] = {0, 2, 1, 0, 4, 3};
    int indptr[5] = {0, 2, 3, 5, 6};
    failures += !run_case("tiny 4x5 scalar path", 4, 5, 6, nnz, col, indptr);
  }

  /* 2. Minimal bulk case: 8 rows, all rowlen 2 -> one SpVV8 group */
  rowlens = (int *)malloc(8 * sizeof(int));
  for (int r = 0; r < 8; r++) rowlens[r] = 2;
  failures += !run_random_case("8 rows uniform len 2 (one bulk group)", 8, 16, rowlens);
  free(rowlens);

  /* 3. Uniform 64x64, rowlen 16: multiple bulk groups over multiple panels */
  rowlens = (int *)malloc(64 * sizeof(int));
  for (int r = 0; r < 64; r++) rowlens[r] = 16;
  failures += !run_random_case("64x64 uniform len 16 (bulk groups)", 64, 64, rowlens);
  free(rowlens);

  /* 4. Mixed lengths 0..40: empty rows, bulk groups, remainders,
     and rows longer than 32 (forced to the scalar path) */
  rowlens = (int *)malloc(200 * sizeof(int));
  for (int r = 0; r < 200; r++) rowlens[r] = rand() % 41;
  failures += !run_random_case("200x300 mixed len 0-40", 200, 300, rowlens);
  free(rowlens);

  /* 5. Rows not a multiple of 8, repeated lengths -> bulk + remainder mix */
  rowlens = (int *)malloc(37 * sizeof(int));
  for (int r = 0; r < 37; r++) rowlens[r] = 1 + (r % 3);
  failures += !run_random_case("37x50 lens 1-3 (bulk + remainder)", 37, 50, rowlens);
  free(rowlens);

  /* 6. Tridiagonal 100x100: takes the banded branch in process() */
  {
    int rows = 100, cols = 100;
    double *nnz = (double *)malloc(3 * rows * sizeof(double));
    int *col = (int *)malloc(3 * rows * sizeof(int));
    int *indptr = (int *)malloc((rows + 1) * sizeof(int));
    indptr[0] = 0;
    int p = 0;
    for (int r = 0; r < rows; r++) {
      for (int c = r - 1; c <= r + 1; c++) {
        if (c < 0 || c >= cols) continue;
        col[p] = c;
        nnz[p++] = rand_val();
      }
      indptr[r + 1] = p;
    }
    failures += !run_case("100x100 tridiagonal (banded)", rows, cols, p, nnz, col, indptr);
    free(nnz); free(col); free(indptr);
  }

  if (failures) {
    printf("%d test case(s) FAILED\n", failures);
    return 1;
  }
  printf("All test cases passed\n");
  return 0;
}
