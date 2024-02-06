#include <cstdio>

#include "image_rotation.h"

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
static float *input_images_gpu, *output_images_gpu;

void rotate_image_native(float *input_images, float *output_images, int W, int H,
                        float sin_theta, float cos_theta, int num_src_images) {
  float x0 = W / 2.0f;
  float y0 = H / 2.0f;

  // Rotate images
  for (int i = 0; i < num_src_images; i++) {
    for (int dest_x = 0; dest_x < W; dest_x++) {
      for (int dest_y = 0; dest_y < H; dest_y++) {
        float xOff = dest_x - x0;
        float yOff = dest_y - y0;
        int src_x = (int) (xOff * cos_theta + yOff * sin_theta + x0);
        int src_y = (int) (yOff * cos_theta - xOff * sin_theta + y0);
        if ((src_x >= 0) && (src_x < W) && (src_y >= 0) && (src_y < H)) {
          output_images[i * H * W + dest_y * W + dest_x] =
              input_images[i * H * W + src_y * W + src_x];
        } else {
          output_images[i * H * W + dest_y * W + dest_x] = 0.0f;
        }
      }
    }
  }
}

__global__ void rotate_image_kernel(float *input_images, float *output_images, int W, int H,
  float sin_theta, float cos_theta, int num_src_images) {
  float x0 = W / 2.0f;
  float y0 = H / 2.0f;

  int i = blockIdx.x / 4;
  int portion = blockIdx.x % 4;
  int dest_y = (portion * H / 4 + threadIdx.x);
  int dest_x = threadIdx.y;
  float xOff = dest_x - x0;
  float yOff = dest_y - y0;
  int src_x = (int) (xOff * cos_theta + yOff * sin_theta + x0);
  int src_y = (int) (yOff * cos_theta - xOff * sin_theta + y0);
  if ((src_x >= 0) && (src_x < W) && (src_y >= 0) && (src_y < H)) {
    output_images[i * H * W + dest_y * W + dest_x] =
    input_images[i * H * W + src_y * W + src_x];
  } else {
    output_images[i * H * W + dest_y * W + dest_x] = 0.0f;
  }
}

__global__ void test() {
  printf("In test funtion\n");
}



void rotate_image(float *input_images, float *output_images, int W, int H,
                  float sin_theta, float cos_theta, int num_src_images) {

  // (TODO) Upload input images to GPU
  CHECK_CUDA(cudaMemcpy(input_images_gpu, input_images, sizeof(float) * W * H * num_src_images, cudaMemcpyHostToDevice));

  // (TODO) Launch kernel on GPU
  dim3 gridDim(num_src_images * 4);
  dim3 blockDim(H / 4, W);
  rotate_image_kernel<<<gridDim, blockDim>>>(input_images_gpu, output_images_gpu, W, H, sin_theta, cos_theta, num_src_images);
  CHECK_CUDA(cudaGetLastError());

  // (TODO) Download output images from GPU
  CHECK_CUDA(cudaMemcpy(output_images, output_images_gpu, sizeof(float) * W * H * num_src_images, cudaMemcpyDeviceToHost));
  printf("output: %f", output_images[0]);

  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void rotate_image_init(int image_width, int image_height, int num_src_images) {
  // (TODO) Allocate device memory
  CHECK_CUDA(cudaMalloc(&input_images_gpu, sizeof(float) * image_height * image_width * num_src_images));
  CHECK_CUDA(cudaMalloc(&output_images_gpu, sizeof(float) * image_height * image_width * num_src_images));

  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void rotate_image_cleanup() {
  // (TODO) Free device memory
  CHECK_CUDA(cudaFree(input_images_gpu));
  CHECK_CUDA(cudaFree(output_images_gpu));

  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
