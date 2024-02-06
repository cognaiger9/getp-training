#include "vec_add.h"

#include <pthread.h>
#include <immintrin.h>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <omp.h>

#define TILESIZE 1024

static int elementPerThread;
static float *A;
static float *B;
static float *C;
static int num_threads;
static int M;

void *sub_add(void *args)
{
  long index = (long)args;
  long base = index * elementPerThread;
  long end = base + elementPerThread;
  if (index == num_threads - 1) {
    end = M;
  }
  __m256 a1, b1, c1;
  for (long i = base; i < end; i += TILESIZE)
  {
    for (long ii = i; ii < std::min(end, i + TILESIZE); ii += 8)
    {
      a1 = _mm256_load_ps(&A[ii]);
      b1 = _mm256_load_ps(&B[ii]);
      c1 = _mm256_add_ps(a1, b1);
      _mm256_store_ps(&C[ii], c1);
    }
  }
  pthread_exit(NULL);
}

void vec_add(float *_A, float *_B, float *_C, int _M, int _num_threads)
{
  // IMPLEMENT HERE
  elementPerThread = _M / _num_threads;
  A = _A;
  B = _B;
  C = _C;
  pthread_t threads[_num_threads];
  for (long i = 0; i < _num_threads; i++)
  {
    int rc = pthread_create(&threads[i], NULL, sub_add, (void *)i);
    if (rc)
    {
      std::cout << "Error, return code is: " << rc << std::endl;
    }
  }

  for (int i = 0; i < _num_threads; i++)
  {
    int rc = pthread_join(threads[i], NULL);
    if (rc)
    {
      std::cout << "Return code from pthread_join() is " << rc << std::endl;
    }
  }
}