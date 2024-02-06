#include <cstdio>
#include <cuda.h>

#include "matmul.h"
#include "util.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)


void naive_cpu_matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        _C[i * N + j] += _A[i * K + k] * _B[k * N + j];
      }
    }
  }
}

#define BLOCKSIZE 32

__global__ void kernel_cpu_matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  // coalescing memory
  int i = blockIdx.y * BLOCKSIZE;                  // starting row on C (one block)
  int j = blockIdx.x * BLOCKSIZE;                  // starting col on C (one block)

  int threadX = threadIdx.x;
  int threadY = threadIdx.y;                // thread in same warp -> same threadOffsetY

  
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  float tmp = 0.0f;
  for (int k = 0; k < K; k += BLOCKSIZE) {
    // Each thread load one portion of A and B into shared memory
    As[threadX + threadY * BLOCKSIZE] = _A[(i + threadY) * K + k + threadX];
    Bs[threadX + threadY * BLOCKSIZE] = _B[(k + threadY) * N + j + threadX];
    __syncthreads();

    // Compute data on shared memory
    for (int kIdx = 0; kIdx < BLOCKSIZE; kIdx++) {
      tmp += As[threadY * BLOCKSIZE + kIdx] * Bs[kIdx * BLOCKSIZE + threadX];
    }
    __syncthreads();
  }
  _C[(i + threadY) * N + j + threadX] = tmp;
}


static int NGPU;
// Device(GPU) pointers
static float *A_gpu[512];
static float *B_gpu[512];
static float *C_gpu[512];

static int Mbegin[512];
static int Mend[512];

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {

  // Create streams, events and allocate memory for each device
  cudaEvent_t events[NGPU];
  cudaStream_t streams[NGPU];
  for (int i = 0; i < NGPU; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamCreate(&streams[i]));
    CHECK_CUDA(cudaEventCreate(&events[i]));
  }

  // H2D transfer
  for (int i = 0; i < NGPU; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMemcpyAsync(B_gpu[i], _B, N * K * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
    CHECK_CUDA(cudaMemcpyAsync(A_gpu[i], &_A[Mbegin[i] * K], (Mend[i] - Mbegin[i]) * K * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
    

    // Launch kernel on a GPU
    dim3 block(BLOCKSIZE, BLOCKSIZE);
    dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE, (Mend[i] - Mbegin[i] + BLOCKSIZE - 1) / BLOCKSIZE);
    kernel_cpu_matmul<<<grid, block, 0, streams[i]>>>(A_gpu[i], B_gpu[i], C_gpu[i], Mend[i] - Mbegin[i], N, K);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaMemcpyAsync(&_C[Mbegin[i] * N], C_gpu[i], (Mend[i] - Mbegin[i]) * N * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
  }

  // Synchronize all GPU
  /*
  for (int i = 0; i < NGPU; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamSynchronize(streams[i]));
  }
  */
  

  for (int i = 0; i < NGPU; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaStreamDestroy(streams[i]));
  }

  // Download C matrix from GPU

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_init(int M, int N, int K) {
  CHECK_CUDA(cudaGetDeviceCount(&NGPU));
  for (int i = 0; i < NGPU; i++) {
    Mbegin[i] = M / NGPU * i;
    Mend[i] = M / NGPU * (i + 1);
  }

  // Allocate device memory
  for (int i = 0; i < NGPU; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaMalloc(&A_gpu[i], sizeof(float) * (Mend[i] - Mbegin[i]) * K));
    CHECK_CUDA(cudaMalloc(&B_gpu[i], sizeof(float) * K * N));
    CHECK_CUDA(cudaMalloc(&C_gpu[i], sizeof(float) * (Mend[i] - Mbegin[i]) * N));
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_cleanup(float *_A, float *_B, float *_C, int M, int N, int K) {
  // Do any post-matmul cleanup work here.
  for (int i = 0; i < NGPU; i++) {
    CHECK_CUDA(cudaSetDevice(i));
    CHECK_CUDA(cudaFree(A_gpu[i]));
    CHECK_CUDA(cudaFree(B_gpu[i]));
    CHECK_CUDA(cudaFree(C_gpu[i]));
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}