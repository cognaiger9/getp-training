#include <cstdio>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

__global__ void pythagoras(int *pa, int *pb, int *pc, int *presult) {
  int a = *pa;
  int b = *pb;
  int c = *pc;

  if ((a * a + b * b) == c * c)
    *presult = 1;
  else
    *presult = 0;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <num 1> <num 2> <num 3>\n", argv[0]);
    return 0;
  }

  int a = atoi(argv[1]);
  int b = atoi(argv[2]);
  int c = atoi(argv[3]);
  int result = 0;

  // TODO: 1. allocate device memory
  int *ptr;
  cudaMalloc(&ptr, sizeof(int) * 4);

  // TODO: 2. copy data to device
  cudaMemcpy(ptr, &a, 4, cudaMemcpyHostToDevice);
  cudaMemcpy(ptr + 1, &b, 4, cudaMemcpyHostToDevice);
  cudaMemcpy(ptr + 2, &c, 4, cudaMemcpyHostToDevice);

  // TODO: 3. launch kernel
  pythagoras<<<1,1>>>(ptr, ptr + 1, ptr + 2, ptr + 3);
  CHECK_CUDA(cudaGetLastError());

  // TODO: 4. copy result back to host
  cudaMemcpy(&result, ptr + 3, 4, cudaMemcpyDeviceToHost);

  if (result) printf("YES\n");
  else printf("NO\n");

  return 0;
}
