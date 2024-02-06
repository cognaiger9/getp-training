/*
Coalesing memory 1700 - 1900 (4960)
each warp process 32 consecutive column of B and 1 row of A -> produce 32 cells C
*/


#include <cstdio>

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

#define BLOCKSIZE 16

// Device(GPU) pointers
static float *A_gpu, *B_gpu, *C_gpu;

void naive_cpu_matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        _C[i * N + j] += _A[i * K + k] * _B[k * N + j];
      }
    }
  }
}

__global__ void kernel_cpu_matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  int i = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  int j = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  float sum = 0.0f;
  if (i < M && j < N) {
    for (int k = 0; k < K; k++) {
      sum += _A[i * K + k] * _B[k * N + j];
    }
    _C[i * N + j] = sum;
  }
}

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {

  // (TODO) Upload A and B matrix to GPU
  CHECK_CUDA(cudaMemcpy(A_gpu, _A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_gpu, _B, sizeof(float) * K * N, cudaMemcpyHostToDevice));

  // (TODO) Launch kernel on a GPU
  dim3 block(BLOCKSIZE * BLOCKSIZE);
  dim3 grid((M + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);
  
  kernel_cpu_matmul<<<grid,block>>>(A_gpu, B_gpu, C_gpu, M, N, K);
  CHECK_CUDA(cudaGetLastError());

  // (TODO) Download C matrix from GPU
  CHECK_CUDA(cudaMemcpy(_C, C_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}









/*
Vectorization 2100 - 2200 (4096)
Take 128 bit from global memory at one instruction
*/
#define BLOCKSIZE 32

__global__ void kernel_cpu_matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  int i = blockIdx.x * BLOCKSIZE + threadIdx.y;
  int j = blockIdx.y * BLOCKSIZE + threadIdx.x;

  float sum = 0.0f;
  if (i < M && j < N) {
    for (int k = 0; k < K; k += 4) {
      float4 a4 = *(float4*)(&_A[i * K + k]);
      for (int l = 0; l < 4; l++) {
        sum += ((float*)(&a4))[l] * _B[(k + l) * N + j];
      }
    }
    _C[i * N + j] = sum;
  }
}












/*
Shared memory 2700
*/
#include <cstdio>

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

#define BLOCKSIZE 32

// Device(GPU) pointers
static float *A_gpu, *B_gpu, *C_gpu;

void naive_cpu_matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < K; k++) {
      for (int j = 0; j < N; j++) {
        _C[i * N + j] += _A[i * K + k] * _B[k * N + j];
      }
    }
  }
}

__global__ void kernel_cpu_matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  const int cRow = blockIdx.x;
  const int cCol = blockIdx.y;

  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  const int threadRow = threadIdx.x / BLOCKSIZE;
  const int threadCol = threadIdx.x % BLOCKSIZE;

  _A += cRow * BLOCKSIZE * K;
  _B += cCol * BLOCKSIZE;
  _C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

  float tmp = 0.0f;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    As[threadRow * BLOCKSIZE + threadCol] = _A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = _B[threadRow * N + threadCol];

    __syncthreads();
    _A += BLOCKSIZE;
    _B += BLOCKSIZE * N;

    for (int dotIdx = 0; dotIdx < BLOCKSIZE; dotIdx++) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    __syncthreads();
  }
  _C[threadRow * N + threadCol] = tmp;
}

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {

  // (TODO) Upload A and B matrix to GPU
  CHECK_CUDA(cudaMemcpy(A_gpu, _A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_gpu, _B, sizeof(float) * K * N, cudaMemcpyHostToDevice));

  // (TODO) Launch kernel on a GPU
  dim3 block(BLOCKSIZE * BLOCKSIZE);
  dim3 grid((M + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);
  
  kernel_cpu_matmul<<<grid,block>>>(A_gpu, B_gpu, C_gpu, M, N, K);
  CHECK_CUDA(cudaGetLastError());

  // (TODO) Download C matrix from GPU
  CHECK_CUDA(cudaMemcpy(_C, C_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}















/*
Coalescing and unrolling ~2200
*/
#define BLOCKSIZE 32
__global__ void kernel_cpu_matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  // coalescing memory
  int i = blockIdx.y * BLOCKSIZE + threadIdx.y;
  int j = blockIdx.x * BLOCKSIZE + threadIdx.x;

  float tmp = 0.0f;
  for (int k = 0; k < K; k += 4) {
    float a1 = _A[i * K + k + 0];
    float a2 = _A[i * K + k + 1];
    float a3 = _A[i * K + k + 2];
    float a4 = _A[i * K + k + 3];

    float b1 = _B[(k + 0) * N + j];
    float b2 = _B[(k + 1) * N + j];
    float b3 = _B[(k + 2) * N + j];
    float b4 = _B[(k + 3) * N + j];

    tmp += a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4;
  }
  _C[i * N + j] = tmp;
}

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {

  // (TODO) Upload A and B matrix to GPU
  CHECK_CUDA(cudaMemcpy(A_gpu, _A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_gpu, _B, sizeof(float) * K * N, cudaMemcpyHostToDevice));

  // (TODO) Launch kernel on a GPU
  dim3 block(BLOCKSIZE, BLOCKSIZE);
  dim3 grid((M + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);
  
  kernel_cpu_matmul<<<grid,block>>>(A_gpu, B_gpu, C_gpu, M, N, K);
  CHECK_CUDA(cudaGetLastError());

  // (TODO) Download C matrix from GPU
  CHECK_CUDA(cudaMemcpy(_C, C_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}












/*

Triple buffer with coalescing 2100

*/
#include <cstdio>

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


// Device(GPU) pointers
static float *A_gpu, *B_gpu, *C_gpu;

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
#define TRANSBLOCK 4

__global__ void kernel_cpu_matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  // coalescing memory
  int i = blockIdx.y * BLOCKSIZE + threadIdx.y;
  int j = blockIdx.x * BLOCKSIZE + threadIdx.x;

  float tmp = 0.0f;
  if (i < M && j < N) {
    for (int k = 0; k < K; k++) {
      tmp += _A[i * K + k] * _B[k * N + j];
    }
    _C[i * N + j] = tmp;
  }
}

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {

  // Divide row of A
  int Mbegin[TRANSBLOCK];
  int Mend[TRANSBLOCK];
  for (int i = 0; i < TRANSBLOCK; i++) {
    Mbegin[i] = M / TRANSBLOCK * i;
    Mend[i] = M / TRANSBLOCK * (i + 1);
  }

  // Create events
  cudaEvent_t events[2 * TRANSBLOCK];
  for (int i = 0; i < 2 * TRANSBLOCK; i++) {
    CHECK_CUDA(cudaEventCreate(&events[i]));
  }

  // Create stream
  cudaStream_t calStream, h2dStream, d2hStream;
  CHECK_CUDA(cudaStreamCreate(&calStream));
  CHECK_CUDA(cudaStreamCreate(&h2dStream));
  CHECK_CUDA(cudaStreamCreate(&d2hStream));

  // Upload A and B matrix to GPU
  CHECK_CUDA(cudaMemcpyAsync(B_gpu, _B, sizeof(float) * K * N, cudaMemcpyHostToDevice, h2dStream));
  for (int i = 0; i < TRANSBLOCK; i++) {
    // Upload portion of matrix A in data stream
    CHECK_CUDA(cudaMemcpyAsync(&A_gpu[Mbegin[i] * K], &_A[Mbegin[i] * K], (Mend[i] - Mbegin[i]) * K * sizeof(float), cudaMemcpyHostToDevice, h2dStream));
    CHECK_CUDA(cudaEventRecord(events[2 * i], h2dStream));

    dim3 block(BLOCKSIZE, BLOCKSIZE);
    dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE, (Mend[i] - Mbegin[i] + BLOCKSIZE - 1) / BLOCKSIZE);

    // Wait for the data on data stream
    CHECK_CUDA(cudaStreamWaitEvent(calStream, events[2 * i]));
    kernel_cpu_matmul<<<grid, block, 0, calStream>>>(&A_gpu[Mbegin[i] * K], B_gpu, &C_gpu[Mbegin[i] * N], (Mend[i] - Mbegin[i]), N, K);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(events[2 * i + 1], calStream));
    CHECK_CUDA(cudaStreamWaitEvent(d2hStream, events[2 * i + 1]));
    CHECK_CUDA(cudaMemcpyAsync(&_C[Mbegin[i] * N], &C_gpu[Mbegin[i] * N], (Mend[i] - Mbegin[i]) * N * sizeof(float), cudaMemcpyDeviceToHost, d2hStream));
  }

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_init(int M, int N, int K) {
  // Allocate device memory
    CHECK_CUDA(cudaMalloc(&A_gpu, sizeof(float) * M * K));
    CHECK_CUDA(cudaMalloc(&B_gpu, sizeof(float) * K * N));
    CHECK_CUDA(cudaMalloc(&C_gpu, sizeof(float) * M * N));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_cleanup(float *_A, float *_B, float *_C, int M, int N, int K) {
  // Do any post-matmul cleanup work here.
    CHECK_CUDA(cudaFree(A_gpu));
    CHECK_CUDA(cudaFree(B_gpu));
    CHECK_CUDA(cudaFree(C_gpu));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}












/*
Shared memory 3200 for 8192 (correct version)
*/
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
  if (i + threadY >= M || j + threadX >= N) {
    return;
  }
  _C[(i + threadY) * N + j + threadX] = tmp;
}

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {

  // Upload A and B matrix to GPU
  CHECK_CUDA(cudaMemcpy(A_gpu, _A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_gpu, _B, sizeof(float) * K * N, cudaMemcpyHostToDevice));

  // Launch kernel on a GPU
  dim3 block(BLOCKSIZE, BLOCKSIZE);
  dim3 grid((N + BLOCKSIZE - 1) / BLOCKSIZE, (M + BLOCKSIZE - 1) / BLOCKSIZE);
  
  kernel_cpu_matmul<<<grid, block>>>(A_gpu, B_gpu, C_gpu, M, N, K);
  CHECK_CUDA(cudaGetLastError());

  // (TODO) Download C matrix from GPU
  CHECK_CUDA(cudaMemcpy(_C, C_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
















/*
4 GPU with shared memory > 10000
*/

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






srun ./main -n 3 4096 4096 4096
srun nsys profile --cudabacktrace=all ./main -n 3 4096 4096 4096
srun ncu -o ncu_report --set full ./main -n 5 4096 4096 4096

TARGET=main
OBJECTS=util.o matmul.o main.o

CPPFLAGS=-std=c++11 -O3 -Wall -march=native -mavx2 -mno-avx512f -mfma -fopenmp
CPPFLAGS+= -I/usr/local/cuda/include/
LDFLAGS=-lm -lcudart -lcublas -lnvToolsExt
LDFLAGS+=-L/usr/local/cuda/lib64
LDLIBS=-lm -lmpi -lmpi_cxx -lnuma

NVCC=/usr/local/cuda/bin/nvcc

all: $(TARGET)

$(TARGET): $(OBJECTS)
	g++ $(CPPFLAGS) $^ -o $@ $(LDFLAGS)

main_mpi: util.o matmul_mpi.o main_mpi.o
	g++ $(CPPFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

matmul.o: matmul.cu
	$(NVCC) -c -o $@ $^

matmul_mpi.o: matmul_mpi.cu
	$(NVCC) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)