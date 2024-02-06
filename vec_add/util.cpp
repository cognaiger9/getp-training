#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>
#include <numa.h>

static double start_time[8];

static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void timer_start(int i) {
    start_time[i] = get_time();
}

double timer_stop(int i) {
    return get_time() - start_time[i];
}

void check_vec_add(float *A, float *B, float *C, int M) {
  printf("Validating...\n");

  float *C_ans;
  alloc_vec(&C_ans, M);
  zero_vec(C_ans, M);
  for (int i = 0; i < M; ++i) {
    C_ans[i] = A[i] + B[i];
  }

  bool is_valid = true;
  int cnt = 0, thr = 10;
  float eps = 1e-3;
  for (int i = 0; i < M; ++i) {
    float c = C[i];
    float c_ans = C_ans[i];
    if (fabsf(c - c_ans) > eps && (c_ans == 0 || fabsf((c - c_ans) / c_ans) > eps)) {
      ++cnt;
      if (cnt <= thr)
        printf("C[%d] : correct_value = %f, your_value = %f\n", i, c_ans, c);
      if (cnt == thr + 1)
        printf("Too many error, only first %d values are printed.\n", thr);
      is_valid = false;
    }
  }

  if (is_valid) {
    printf("Result: VALID\n");
  } else {
    printf("Result: INVALID\n");
  }
}

void print_vec(float *m, int R) {
  for (int i = 0; i < R; ++i) { 
    printf("%+.3f ", m[i]);
  }
  printf("\n");
}

void alloc_vec(float **m, int R) {
  struct bitmask* nodemask = numa_allocate_nodemask();
  numa_bitmask_setbit(nodemask, 0);
  numa_bitmask_setbit(nodemask, 1);
  *m = (float *)numa_alloc_interleaved(sizeof(float) * R);
  numa_free_nodemask(nodemask);
  if (*m == NULL) {
    printf("Failed to allocate memory for vector.\n");
    exit(0);
  }
}

void rand_vec(float *m, int R) {
  for (int i = 0; i < R; i++) { 
    m[i] = (float) rand() / RAND_MAX - 0.5;
  }
}

void zero_vec(float *m, int R) {
  memset(m, 0, sizeof(float) * R);
}
