/*
First valid gpu 3 kernel 500 GFLOPS
*/
#include <cstdlib>
#include <cstdio>

#include "convolution.cuh"

#define CHECK_CUDA(call)                                                 \
  do                                                                     \
  {                                                                      \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess)                                          \
    {                                                                    \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

void naive_cpu_im2col(half *_I, half *workspace, int N, int C, int H, int W,
                      int R, int S, int pad_h, int pad_w, int stride_h,
                      int stride_w, int dilation_h, int dilation_w)
{

  // Naive CPU im2col
  const int ON = N;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  for (int on = 0; on < ON; ++on)
  {
    for (int oh = 0; oh < OH; ++oh)
    {
      for (int ow = 0; ow < OW; ++ow)
      {
        for (int c = 0; c < C; ++c)
        {
          for (int r = 0; r < R; ++r)
          {
            for (int s = 0; s < S; ++s)
            {
              const int n = on;
              const int h = oh * stride_h - pad_h + r * dilation_h;
              const int w = ow * stride_w - pad_w + s * dilation_w;

              if (h < 0 || h >= H || w < 0 || w >= W)
                continue;

              workspace[((c * R * S) + (r * S) + s) * (ON * OH * OW) +
                        (on * OH * OW + oh * OW + ow)] =
                  _I[n * C * H * W + c * H * W + h * W + w];
            }
          }
        }
      }
    }
  }
}

void naive_cpu_matmul_TC(half *_A, half *_B, half *_C, int M, int N, int K)
{

  for (int i = 0; i < M; i++)
  {
    for (int k = 0; k < K; k++)
    {
      for (int j = 0; j < N; j++)
      {
        _C[i * N + j] = (half)((float)(_C[i * N + j]) + (float)_A[i * K + k] * (float)_B[k * N + j]);
      }
    }
  }
}

void reshape(half *_src, half *_dst, int N, int K, int OH, int OW)
{
  size_t chunk = OH * OW;

  for (int on = 0; on < N; ++on)
  {
    for (int k = 0; k < K; ++k)
    {
      memcpy((void *)(_dst + ((on * K + k) * chunk)),
             (void *)(_src + ((k * N + on) * chunk)), chunk * sizeof(half));
    }
  }
}

void naive_cpu_convolution_im2col(half *_I, half *_F, half *_O, half *_BUF1,
                                  half *_BUF2, int N, int C, int H, int W,
                                  int K, int R, int S, int pad_h, int pad_w,
                                  int stride_h, int stride_w, int dilation_h,
                                  int dilation_w)
{
  half *I = _I, *F = _F, *O = _O, *BUF1 = _BUF1, *BUF2 = _BUF2;

  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  naive_cpu_im2col(I, BUF1, N, C, H, W, R, S, pad_h, pad_w, stride_h, stride_w,
                   dilation_h, dilation_w);

  naive_cpu_matmul_TC(F, BUF1, BUF2, K, N * OH * OW, C * R * S);

  reshape(BUF2, O, N, K, OH, OW);
}


static half *I_GPU;
static half *F_GPU;
static half *O_GPU;
static half *BUF1_GPU;
static half *BUF2_GPU;
#define BLOCKSIZE 512

__global__ void kernel_gpu_im2col(half *_I, half *workspace, int N, int C, int H, int W,
                                  int R, int S, int pad_h, int pad_w, int stride_h,
                                  int stride_w, int dilation_h, int dilation_w)
{
  half *I = _I;

  // Kernel CPU im2col
  const int ON = N;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  
  long global = blockIdx.x * blockDim.x + threadIdx.x;
  int on = global / (OH * OW * C * R * S);
  global = global % (OH * OW * C * R * S);
  int oh = global / (OW * C * R * S);
  global = global % (OW * C * R * S);
  int ow = global / (C * R * S);
  global = global % (C * R * S);
  int c = global / (R * S);
  global = global % (R * S);
  int r = global / S;
  int s = global % S;

  if (on >= ON || oh >= OH || ow >= OW || c >= C || r >= R || s >= S)
  {
    return;
  }

  const int n = on;
  const int h = oh * stride_h - pad_h + r * dilation_h;
  const int w = ow * stride_w - pad_w + s * dilation_w;

  if (h < 0 || h >= H || w < 0 || w >= W)
    return;

  workspace[((c * R * S) + (r * S) + s) * (ON * OH * OW) + (on * OH * OW + oh * OW + ow)] = I[n * C * H * W + c * H * W + h * W + w];
  

  
}

__global__ void kernel_gpu_matmul(half *_A, half *_B, half *_C, int M, int N, int K)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= M || j >= N)
  {
    return;
  }
  half tmp = 0;
  for (int k = 0; k < K; k++)
  {
    tmp = (half)((float)tmp + (float)_A[i * K + k] * (float)_B[k * N + j]);
  }
  _C[i * N + j] = tmp;
}

__global__ void kernel_reshape(half *_src, half *_dst, int N, int K, int OH, int OW)
{
  size_t chunk = OH * OW;

  int on = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  if (on >= N || k >= K)
  {
    return;
  }

  memcpy((void *)(_dst + ((on * K + k) * chunk)),
         (void *)(_src + ((k * N + on) * chunk)), chunk * sizeof(half));
}


void convolution(half *_I, half *_F, half *_O, half *_BUF1, half *_BUF2, int N,
                 int C, int H, int W, int K, int R, int S, int pad_h, int pad_w,
                 int stride_h, int stride_w, int dilation_h, int dilation_w)
{
  half *I = _I, *F = _F, *O = _O, *BUF1 = _BUF1, *BUF2 = _BUF2;
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  CHECK_CUDA(cudaMemcpy(I_GPU, _I, N * C * H * W * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F_GPU, _F, K * C * R * S * sizeof(half), cudaMemcpyHostToDevice));
  dim3 block(BLOCKSIZE);
  dim3 grid((N * K * OH * OW + BLOCKSIZE - 1) / BLOCKSIZE);

  // Start kernel call
  dim3 block1(1024);
  dim3 grid1((ON * OH * OW * C * R * S + block1.x - 1) / block1.x);
  kernel_gpu_im2col<<<grid1, block1>>>(I_GPU, BUF1_GPU, N, C, H, W, R, S, pad_h, pad_w, stride_h, stride_w,
                                       dilation_h, dilation_w);
  CHECK_CUDA(cudaGetLastError());

  dim3 block2(32, 32);
  dim3 grid2((N * OH * OW + block2.x - 1) / block2.x, (K + block2.y - 1) / block2.y);
  kernel_gpu_matmul<<<grid2, block2>>>(F_GPU, BUF1_GPU, BUF2_GPU, K, N * OH * OW, C * R * S);
  CHECK_CUDA(cudaGetLastError());

  dim3 block3(32, 32);
  dim3 grid3((N + block3.x - 1) / block3.x, (K + block3.y - 1) / block3.y);
  kernel_reshape<<<grid3, block3>>>(BUF2_GPU, O_GPU, N, K, OH, OW);
  CHECK_CUDA(cudaGetLastError());

  CHECK_CUDA(cudaMemcpy(O, O_GPU, ON * OC * OH * OW * sizeof(half), cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_initialize(int N, int C, int H, int W, int K, int R, int S,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w)
{
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  CHECK_CUDA(cudaMalloc(&I_GPU, N * C * H * W * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&F_GPU, K * C * R * S * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&O_GPU, ON * OC * OH * OW * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&BUF1_GPU, C * R * S * N * OH * OW * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&BUF2_GPU, K * N * OH * OW * sizeof(half)));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_cleanup(half *_I, half *_F, half *_O, int N, int C, int H,
                         int W, int K, int R, int S, int pad_h, int pad_w,
                         int stride_h, int stride_w, int dilation_h,
                         int dilation_w)
{
  CHECK_CUDA(cudaFree(I_GPU));
  CHECK_CUDA(cudaFree(F_GPU));
  CHECK_CUDA(cudaFree(O_GPU));
  CHECK_CUDA(cudaFree(BUF1_GPU));
  CHECK_CUDA(cudaFree(BUF2_GPU));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}








/*
Coaslescing
*/
__global__ void kernel_gpu_im2col(half *_I, half *workspace, int N, int C, int H, int W,
                                  int R, int S, int pad_h, int pad_w, int stride_h,
                                  int stride_w, int dilation_h, int dilation_w)
{
  half *I = _I;

  // Kernel GPU im2col
  const int ON = N;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  
  long global = blockIdx.x * blockDim.x + threadIdx.x;
  int c = global / (ON * OH * OW * R * S);
  global = global % (ON * OH * OW * R * S);
  int r = global / (ON * OH * OW * S);
  global = global % (ON * OH * OW * S);
  int s = global / (ON * OH * OW);
  global = global % (ON * OH * OW);
  int on = global / (OH * OW);
  global = global % (OH * OW);
  int oh = global / OW;
  int ow = global % OW;

  if (on >= ON || oh >= OH || ow >= OW || c >= C || r >= R || s >= S)
  {
    return;
  }

  const int n = on;
  const int h = oh * stride_h - pad_h + r * dilation_h;
  const int w = ow * stride_w - pad_w + s * dilation_w;

  if (h < 0 || h >= H || w < 0 || w >= W)
    return;

  workspace[((c * R * S) + (r * S) + s) * (ON * OH * OW) + (on * OH * OW + oh * OW + ow)] = I[n * C * H * W + c * H * W + h * W + w];
}

__global__ void kernel_gpu_matmul(half *_A, half *_B, half *_C, int M, int N, int K)
{
  // coalescing memory
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= M || j >= N)
  {
    return;
  }
  half tmp = 0;
  for (int k = 0; k < K; k++)
  {
    tmp = (half)((float)tmp + (float)_A[i * K + k] * (float)_B[k * N + j]);
  }
  _C[i * N + j] = tmp;
}
















srun ./main -n 5 -v 3 5 11 33 34 3 4 7 1 3 3 3
srun ./main -n 5 -v 3 5 11 17 55 3 4 7 1 3 3 3

srun ./main -n 5 -v 3 8 16 32 32 3 8 8
srun ./main -n 5 -v 3 8 11 33 34 3 5 8
srun ./main -n 5 -v 3 8 16 256 256 3 8 8
srun ./main -n 5 -v 3 8 256 1024 256 3 8 8
srun ./main -n 5 3 8 512 1024 512 3 8 8 (performance)

srun nsys profile --cudabacktrace=all ./main 8 128 128 128 3 8 8