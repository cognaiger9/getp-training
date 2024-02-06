#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define NUM_ELEM 1024 * 128

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

double print_result(float c, double t) {
  printf("Result : %f\n", c);
  printf("N : %d\n", NUM_ELEM);
  printf("Elapsed time : %f sec\n", t);
  printf("Throughput : %.5f GFLOPS\n\n", 2 * NUM_ELEM / t / 1e9);
}

void naive(float *a, float *b) {
  float c = 0.0;
  double st = get_time();
  for (unsigned int i = 0; i < NUM_ELEM; ++i) {
    c += a[i] * b[i];
  }
  double et = get_time();
  print_result(c, et - st);
}

void unroll(float *a, float *b) {
  float s0, s1, s2, s3, s4, s5, s6, s7;
  s0 = s1 = s2 = s3 = s4 = s5 = s6 = s7 = 0.0;

  double st = get_time();
  for (unsigned int i = 0; i < NUM_ELEM / 8; ++i) {
    s0 += a[8 * i + 0] * b[8 * i + 0];
    s1 += a[8 * i + 1] * b[8 * i + 1];
    s2 += a[8 * i + 2] * b[8 * i + 2];
    s3 += a[8 * i + 3] * b[8 * i + 3];
    s4 += a[8 * i + 4] * b[8 * i + 4];
    s5 += a[8 * i + 5] * b[8 * i + 5];
    s6 += a[8 * i + 6] * b[8 * i + 6];
    s7 += a[8 * i + 7] * b[8 * i + 7];
  }
  float c = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
  double et = get_time();
  print_result(c, et - st);
}

void vector(float *a, float *b) {
  __m256 a0;
  __m256 b0;
  __m256 s0;
  s0 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);

  double st = get_time();
  for (unsigned int i = 0; i < NUM_ELEM / 8; ++i) {
    a0 = _mm256_load_ps(a + i * 8);
    b0 = _mm256_load_ps(b + i * 8);
    s0 = _mm256_fmadd_ps(a0, b0, s0);
  }

  float c = s0[0] + s0[1] + s0[2] + s0[3] + s0[4] + s0[5] + s0[6] + s0[7];
  double et = get_time();
  print_result(c, et - st);
}

void unroll_vector(float *a, float *b) {
  __m256 a0, a1, a2, a3, a4, a5, a6, a7;
  __m256 b0, b1, b2, b3, b4, b5, b6, b7;
  __m256 s0, s1, s2, s3, s4, s5, s6, s7;
  s0 = s1 = s2 = s3 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);
  s4 = s5 = s6 = s7 = _mm256_set_ps(0., 0., 0., 0., 0., 0., 0., 0.);

  double st = get_time();
  for (unsigned int i = 0; i < NUM_ELEM / (8 * 8); ++i) {
    a0 = _mm256_load_ps(a + i * 64);
    b0 = _mm256_load_ps(b + i * 64);
    a1 = _mm256_load_ps(a + i * 64 + 8);
    b1 = _mm256_load_ps(b + i * 64 + 8);
    a2 = _mm256_load_ps(a + i * 64 + 16);
    b2 = _mm256_load_ps(b + i * 64 + 16);
    a3 = _mm256_load_ps(a + i * 64 + 24);
    b3 = _mm256_load_ps(b + i * 64 + 24);
    a4 = _mm256_load_ps(a + i * 64 + 32);
    b4 = _mm256_load_ps(b + i * 64 + 32);
    a5 = _mm256_load_ps(a + i * 64 + 40);
    b5 = _mm256_load_ps(b + i * 64 + 40);
    a6 = _mm256_load_ps(a + i * 64 + 48);
    b6 = _mm256_load_ps(b + i * 64 + 48);
    a7 = _mm256_load_ps(a + i * 64 + 56);
    b7 = _mm256_load_ps(b + i * 64 + 56);

    s0 = _mm256_fmadd_ps(a0, b0, s0);
    s1 = _mm256_fmadd_ps(a1, b1, s1);
    s2 = _mm256_fmadd_ps(a2, b2, s2);
    s3 = _mm256_fmadd_ps(a3, b3, s3);
    s4 = _mm256_fmadd_ps(a4, b4, s4);
    s5 = _mm256_fmadd_ps(a5, b5, s5);
    s6 = _mm256_fmadd_ps(a6, b6, s6);
    s7 = _mm256_fmadd_ps(a7, b7, s7);
  }

  float c0 = s0[0] + s0[1] + s0[2] + s0[3] + s0[4] + s0[5] + s0[6] + s0[7];
  float c1 = s1[0] + s1[1] + s1[2] + s1[3] + s1[4] + s1[5] + s1[6] + s1[7];
  float c2 = s2[0] + s2[1] + s2[2] + s2[3] + s2[4] + s2[5] + s2[6] + s2[7];
  float c3 = s3[0] + s3[1] + s3[2] + s3[3] + s3[4] + s3[5] + s3[6] + s3[7];
  float c4 = s4[0] + s4[1] + s4[2] + s4[3] + s4[4] + s4[5] + s4[6] + s4[7];
  float c5 = s5[0] + s5[1] + s5[2] + s5[3] + s5[4] + s5[5] + s5[6] + s5[7];
  float c6 = s6[0] + s6[1] + s6[2] + s6[3] + s6[4] + s6[5] + s6[6] + s6[7];
  float c7 = s7[0] + s7[1] + s7[2] + s7[3] + s7[4] + s7[5] + s7[6] + s7[7];
  float c = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
  double et = get_time();
  print_result(c, et - st);
}

int main() {
  struct timeval rt;
  gettimeofday(&rt, 0);
  srand(rt.tv_sec);

  float *a, *b;

  a = (float *)aligned_alloc(32, NUM_ELEM * sizeof(float));
  b = (float *)aligned_alloc(32, NUM_ELEM * sizeof(float));

  for (unsigned int i = 0; i < NUM_ELEM; ++i) {
    a[i] = (float)rand() / (float)RAND_MAX;
    b[i] = (float)rand() / (float)RAND_MAX;
  }

  printf("========= Naive =========\n");
  naive(a, b);
  printf("========= Loop Unrolling =========\n");
  unroll(a, b);
  printf("========= Vector Instruction(FMA) =========\n");
  vector(a, b);
  printf("========= Loop Unrolling + Vector Instruction(FMA) =========\n");
  unroll_vector(a, b);
}