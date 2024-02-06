/*
OPENMP device construct
*/

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
  #pragma omp target map(to: _A[0:_M], _B[0:_M]) map(from: _C[0:_M]) num_teams(_num_threads)
  {
    #pragma omp for nowait
    for (int i = 0; i < _M; i++) {
      _C[i] = _B[i] + _A[i];
    }
  }
}