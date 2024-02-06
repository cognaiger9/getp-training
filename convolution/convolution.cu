#include <cstdio>

#include "convolution.h"

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

void naive_cpu_convolution(float *_I, float *_F, float *_O, int N, int C, int H,
                           int W, int K, int R, int S, int pad_h, int pad_w,
                           int stride_h, int stride_w, int dilation_h,
                           int dilation_w)
{
  float *I = _I, *F = _F, *O = _O;
  // Naive CPU convolution
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  // Number of output images
  for (int on = 0; on < ON; ++on)
  {
    // z dimension of each input
    for (int oc = 0; oc < OC; ++oc)
    {
      // h dimension of each image
      for (int oh = 0; oh < OH; ++oh)
      {
        // w dimension of each image
        for (int ow = 0; ow < OW; ++ow)
        {
          float sum = 0;
          for (int c = 0; c < C; ++c)
          {
            for (int r = 0; r < R; ++r)
            {
              for (int s = 0; s < S; ++s)
              {
                const int n = on;
                const int h = oh * stride_h - pad_h + r * dilation_h;
                const int w = ow * stride_w - pad_w + s * dilation_w;
                const int k = oc;
                if (h < 0 || h >= H || w < 0 || w >= W)
                  continue;
                sum += I[((n * C + c) * H + h) * W + w] *
                       F[((k * C + c) * R + r) * S + s];
              }
            }
          }
          O[((on * OC + oc) * OH + oh) * OW + ow] = sum;
        }
      }
    }
  }
}

static float *I_GPU;
static float *F_GPU;
static float *O_GPU;
#define BLOCKSIZE 512

__global__ void kernel_cpu_convolution(float *_I, float *_F, float *_O, int N, int C, int H,
                                       int W, int K, int R, int S, int pad_h, int pad_w,
                                       int stride_h, int stride_w, int dilation_h,
                                       int dilation_w)
{
  float *I = _I, *F = _F, *O = _O;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  int threadId = (blockIdx.x * blockDim.x + threadIdx.x);
  int on = threadId / (OC * OH * OW);
  threadId = threadId % (OC * OH * OW);
  int oc = threadId / (OH * OW);
  threadId = threadId % (OH * OW);
  int oh = threadId / OW;
  int ow = threadId % OW;

  float sum = 0.0f;
  for (int c = 0; c < C; ++c)
  {
    for (int r = 0; r < R; ++r)
    {
      for (int s = 0; s < S; ++s)
      {
        const int n = on;
        const int h = oh * stride_h - pad_h + r * dilation_h;
        const int w = ow * stride_w - pad_w + s * dilation_w;
        const int k = oc;
        if (h < 0 || h >= H || w < 0 || w >= W)
          continue;
        sum += I[((n * C + c) * H + h) * W + w] *
               F[((k * C + c) * R + r) * S + s];
      }
    }
  }
  O[((on * OC + oc) * OH + oh) * OW + ow] = sum;
}

void convolution(float *_I, float *_F, float *_O, int N, int C, int H, int W,
                 int K, int R, int S, int pad_h, int pad_w, int stride_h,
                 int stride_w, int dilation_h, int dilation_w)
{
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  CHECK_CUDA(cudaMemcpy(I_GPU, _I, N * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(F_GPU, _F, K * C * R * S * sizeof(float), cudaMemcpyHostToDevice));
  dim3 block(BLOCKSIZE);
  dim3 grid((N * K * OH * OW + BLOCKSIZE - 1) / BLOCKSIZE);
  kernel_cpu_convolution<<<grid, block>>>(I_GPU, F_GPU, O_GPU, N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
                                          stride_w, dilation_h, dilation_w);

  CHECK_CUDA(cudaMemcpy(_O, O_GPU, N * K * OH * OW * sizeof(float), cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_initialize(int N, int C, int H, int W, int K, int R, int S,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w)
{
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  CHECK_CUDA(cudaMalloc(&I_GPU, N * C * H * W * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&F_GPU, K * C * R * S * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&O_GPU, N * K * OH * OW * sizeof(float)));
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_cleanup(float *_I, float *_F, float *_O, int N, int C, int H,
                         int W, int K, int R, int S, int pad_h, int pad_w,
                         int stride_h, int stride_w, int dilation_h,
                         int dilation_w)
{
  CHECK_CUDA(cudaFree(I_GPU));
  CHECK_CUDA(cudaFree(F_GPU));
  CHECK_CUDA(cudaFree(O_GPU));
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}