/*
600 GFLOPS no SIMD pthread
*/
#include "mat_mul.h"

#include <pthread.h>
#include <immintrin.h>
#include <algorithm>
#include <iostream>

#define MTILESIZE (32)
#define NTILESIZE (512)
#define KTILESIZE (1024)

static float *A;
static float *B;
static float *C;
static long M;
static long N;
static long K;
static long nums_thread;

void *mat_mul_thread(void *args)
{
  long my_args = (long)args;
  long startM = M / nums_thread * my_args + std::min(my_args, M % nums_thread);
  long endM = M / nums_thread * (my_args + 1) + std::min(my_args + 1, M % nums_thread);
  for (long kk = 0; kk < K; kk += KTILESIZE)
  {
    long KTileBoundary1 = std::min(kk + KTILESIZE, K);
    for (long mm = startM; mm < endM; mm += MTILESIZE)
    {
      long MTileBoundary = std::min(mm + MTILESIZE, endM);
      for (long nn = 0; nn < N; nn += NTILESIZE)
      {
        long NTileBoundary = std::min(nn + NTILESIZE, N);

        for (int k = kk; k < KTileBoundary1; k += 4)
        {
          for (int m = mm; m < MTileBoundary; m++)
          {
            float a1 = A[m * K + k + 0];
            float a2 = A[m * K + k + 1];
            float a3 = A[m * K + k + 2];
            float a4 = A[m * K + k + 3];

            for (int n = nn; n < NTileBoundary; n++)
            {
              float b1 = B[(k + 0) * N + n];
              float b2 = B[(k + 1) * N + n];
              float b3 = B[(k + 2) * N + n];
              float b4 = B[(k + 3) * N + n];

              C[m * N + n] += a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4;
            }
          }
        }
      }
    }
  }
  pthread_exit(NULL);
}

void mat_mul(float *_A, float *_B, float *_C, int _M, int _N, int _K, int _num_threads)
{
  // IMPLEMENT HERE
  A = _A;
  B = _B;
  C = _C;
  M = _M;
  N = _N;
  K = _K;
  nums_thread = _num_threads;
  pthread_t threads[_num_threads];
  pthread_attr_t attr[_num_threads];
  cpu_set_t cpus[_num_threads];
  for (long i = 0; i < _num_threads; i++)
  {
    pthread_attr_init(&attr[i]);
    CPU_ZERO(&cpus[i]);
    CPU_SET(i, &cpus[i]);
    pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpus[i]);
    pthread_create(&threads[i], NULL, mat_mul_thread, (void *)i);
  }
  for (int i = 0; i < _num_threads; i++)
  {
    pthread_join(threads[i], NULL);
  }
}