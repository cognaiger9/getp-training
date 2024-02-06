#include "mat_mul.h"

#include <pthread.h>
#include <immintrin.h>
#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <math.h>

#define MTILESIZE (32)
#define NTILESIZE (1024)
#define KTILESIZE (384)

static float *A;
static float *B;
static float *C;
static int M;
static int N;
static int K;
static int nums_thread;

void *mat_mul_thread(void *args)
{
  long long_args = (long)args;
  int my_args = long_args;
  int startM = M / nums_thread * my_args + std::min(my_args, M % nums_thread);
  int endM = M / nums_thread * (my_args + 1) + std::min(my_args + 1, M % nums_thread);
  int evenK = K - K % KTILESIZE;
  for (int kk = 0; kk < evenK; kk += KTILESIZE)
  {
    int KTileBoundary1 = kk + KTILESIZE;
    for (int mm = startM; mm < endM; mm += MTILESIZE)
    {
      int MTileBoundary = std::min(mm + MTILESIZE, endM);
      for (int nn = 0; nn < N; nn += NTILESIZE)
      {
        int NTileBoundary = std::min(nn + NTILESIZE, N);

        for (int k = kk; k < KTileBoundary1; k += 6)
        {
          for (int m = mm; m < MTileBoundary; m++)
          {
            float a1 = A[m * K + k + 0];
            float a2 = A[m * K + k + 1];
            float a3 = A[m * K + k + 2];
            float a4 = A[m * K + k + 3];
            float a5 = A[m * K + k + 4];
            float a6 = A[m * K + k + 5];

            for (int n = nn; n < NTileBoundary; n++)
            {
              float b1 = B[(k + 0) * N + n];
              float b2 = B[(k + 1) * N + n];
              float b3 = B[(k + 2) * N + n];
              float b4 = B[(k + 3) * N + n];
              float b5 = B[(k + 4) * N + n];
              float b6 = B[(k + 5) * N + n];

              C[m * N + n] += a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6;
            }
          }
        }
      }
    }
  }

  for (int mm = startM; mm < endM; mm += MTILESIZE)
  {
    int MTileBoundary = std::min(mm + MTILESIZE, endM);
    for (int nn = 0; nn < N; nn += NTILESIZE)
    {
      int NTileBoundary = std::min(nn + NTILESIZE, N);

      for (int k = evenK; k < K; k += 4)
      {
        for (int m = mm; m < MTileBoundary; m++)
        {
          float a1 = A[m * K + k + 0];
          float a2 = (k + 1 >= K) ? 0 : A[m * K + k + 1];
          float a3 = (k + 2 >= K) ? 0 : A[m * K + k + 2];
          float a4 = (k + 3 >= K) ? 0 : A[m * K + k + 3];

          for (int n = nn; n < NTileBoundary; n++)
          {
            float b1 = B[(k + 0) * N + n];
            float b2 = (k + 1 >= K) ? 0 : B[(k + 1) * N + n];
            float b3 = (k + 2 >= K) ? 0 : B[(k + 2) * N + n];
            float b4 = (k + 3 >= K) ? 0 : B[(k + 3) * N + n];

            C[m * N + n] += a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4;
          }
        }
      }
    }
  }
  pthread_exit(NULL);
}

void mat_mul(float *_A, float *_B, float *_C, int _M, int _N, int _K, int _num_threads)
{
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

    // Limit thread to run only on 1 core
    pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpus[i]);
    pthread_create(&threads[i], &attr[i], mat_mul_thread, (void *)i);
  }

  for (int i = 0; i < _num_threads; i++)
  {
    pthread_join(threads[i], NULL);
  }
}