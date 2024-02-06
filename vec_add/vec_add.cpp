#include "vec_add.h"

#include <pthread.h>
#include <immintrin.h>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include <numa.h>
#include <sched.h>

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
  
  if (index == num_threads - 1)
  {
    end = M;
  }
  for (int i = base; i < end; i++) {
    C[i] = A[i] + B[i];
  }
  pthread_exit(NULL);
}

void vec_add(float *_A, float *_B, float *_C, int _M, int _num_threads)
{
  elementPerThread = _M / _num_threads;
  A = _A;
  B = _B;
  C = _C;
  pthread_t threads[_num_threads];
  pthread_attr_t attr[_num_threads];
  cpu_set_t cpuset[_num_threads];

  for (long i = 0; i < _num_threads; i++)
  {
    pthread_attr_init(&attr[i]);
    CPU_ZERO(&cpuset[i]);
    CPU_SET(i, &cpuset[i]);
    pthread_attr_setaffinity_np(&attr[i], sizeof(cpu_set_t), &cpuset[i]);
    pthread_create(&threads[i], &attr[i], sub_add, (void *)i);
  }

  for (int i = 0; i < _num_threads; i++)
  {
    pthread_join(threads[i], NULL);
  }
}
