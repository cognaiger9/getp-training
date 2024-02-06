#include "matmul.h"
#include <cstdio>
#include <mma.h>

using namespace nvcuda;

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t status_ = call;                                                \
    if (status_ != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(status_));                                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32
#define NUM_WARP ((WMMA_M * WMMA_N) / (WARP_SIZE))
#define C_LAYOUT wmma::mem_row_major

static __global__ void matmul_kernel(half *A, half *B, float *C, int M, int N,
                                     int K) {
  int gj = blockIdx.x;
  int gi = blockIdx.y;
  if (gi * WARP_SIZE >= M || gj * WARP_SIZE >= N)
    return; // boundary check
  int lj = threadIdx.x;
  int li = threadIdx.y;
  int warpId = li;

  __shared__ half Alocal[WARP_SIZE * WARP_SIZE];
  __shared__ half Blocal[WARP_SIZE * WARP_SIZE];

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  int A_row_index = (gi * WARP_SIZE + li);
  int B_col_index = (gj * WARP_SIZE + lj);

  for (int bk = 0; bk < K; bk += WARP_SIZE) {

    for (int offset = 0; offset < NUM_WARP; ++offset) {
      int A_col_index = bk + lj;
      Alocal[(li + offset * blockDim.y) * WARP_SIZE + lj] =
          ((A_row_index + offset * blockDim.y) < M && A_col_index < K)
              ? A[(A_row_index + offset * blockDim.y) * K + A_col_index]
              : (half)(0.0);

      int B_row_index = bk + li + (offset * blockDim.y);
      Blocal[(li + offset * blockDim.y) * WARP_SIZE + lj] =
          (B_row_index < K && B_col_index < N)
              ? B[B_row_index * N + B_col_index]
              : (half)(0.0);
    }
    __syncthreads();

    for (int i = 0; i < WARP_SIZE; i += WMMA_K) {
      int aCol = i;
      int aRow = (warpId / 2) * WMMA_M;
      int bCol = (warpId % 2) * WMMA_N;
      int bRow = i;

      wmma::load_matrix_sync(a_frag, Alocal + aCol + aRow * WARP_SIZE,
                             WARP_SIZE);
      wmma::load_matrix_sync(b_frag, Blocal + bCol + bRow * WARP_SIZE,
                             WARP_SIZE);

      wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    __syncthreads();
  }

  int cRow = (warpId / 2) * WMMA_M + blockIdx.y * blockDim.y * NUM_WARP;
  int cCol = (warpId % 2) * WMMA_N + blockIdx.x * blockDim.x;

  if (cRow + WMMA_M <= M && cCol + WMMA_N <= N) {
    wmma::store_matrix_sync(C + cCol + cRow * N, c_frag, N, C_LAYOUT);
  }
}

static half *A_gpu, *B_gpu;
static float *C_gpu;

void matmul(half *_A, half *_B, float *_C, int M, int N, int K) {
  CHECK_CUDA(
      cudaMemcpy(A_gpu, _A, M * K * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(B_gpu, _B, K * N * sizeof(half), cudaMemcpyHostToDevice));
  dim3 blockDim(WARP_SIZE, 4);
  dim3 gridDim((N + WARP_SIZE - 1) / WARP_SIZE,
               (M + WARP_SIZE - 1) / WARP_SIZE);
  matmul_kernel<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, M, N, K);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(
      cudaMemcpy(_C, C_gpu, M * N * sizeof(float), cudaMemcpyDeviceToHost));
}

void matmul_init(int M, int N, int K) {
  CHECK_CUDA(cudaMalloc(&A_gpu, M * K * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&B_gpu, K * N * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&C_gpu, M * N * sizeof(float)));
}

void matmul_cleanup(half *_A, half *_B, float *_C, int M, int N, int K) {
  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));
}
