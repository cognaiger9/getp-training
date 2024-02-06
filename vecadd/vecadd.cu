#include <cstdio>

#include "vecadd.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

__global__ void vecadd_kernel(const int N, const float *a, const float *b, float *c) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tidx < N) {
    c[tidx] = a[tidx] + b[tidx];
  }
}

// Device(GPU) pointers
static float *A_gpu, *B_gpu, *C_gpu;
#define BLOCKSIZE 1024

void vecadd(float *_A, float *_B, float *_C, int N) {
  // (TODO) Upload A and B vector to GPU
  vecadd_init(N);
  cudaMemcpy(A_gpu, _A, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(B_gpu, _B, sizeof(float) * N, cudaMemcpyHostToDevice);

  // Launch kernel on a GPU
  dim3 blockDim(BLOCKSIZE);
  dim3 gridDim((N + blockDim.x - 1) / blockDim.x);
  vecadd_kernel<<<gridDim, blockDim>>>(N, A_gpu, B_gpu, C_gpu);

  // (TODO) Download C vector from GPU
  cudaMemcpy(_C, C_gpu, sizeof(float) * N, cudaMemcpyDeviceToHost);

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void vecadd_init(int N) {
  // (TODO) Allocate device memory
  cudaMalloc(&A_gpu, N * sizeof(float));
  cudaMalloc(&B_gpu, N * sizeof(float));
  cudaMalloc(&C_gpu, N * sizeof(float));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void vecadd_cleanup(float *_A, float *_B, float *_C, int N) {
  // (TODO) Do any post-vecadd cleanup work here.
  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
