#include "vec_add.h"

#include <pthread.h>
#include <immintrin.h>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <omp.h>

#define TILESIZE 1024

void vec_add(float *_A, float *_B, float *_C, int _M, int _num_threads)
{
  // IMPLEMENT HERE
  #pragma omp parallel _num_threads(_num_threads)
  {
    std::cout << "Number of threads: " << omp_get_num_threads() << ", id: " << omp_get_thread_num() << std::endl;
    #pragma omp for nowait
    for (int i = 0; i < _M; i++) {
      _C[i] = _B[i] + _A[i];
    }
  }
}