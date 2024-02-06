#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define CHECK_OPENCL(err)                                         \
  if (err != CL_SUCCESS)                                          \
  {                                                               \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE);                                           \
  }

int main()
{
  cl_int err;

  /*
You will use the following APIs:
clGetPlatformIDs
clGetPlatformInfo
clGetDeviceIDs
clGetDeviceInfo

[Example output]
Number of platforms: 1
platform: 0
- CL_PLATFORM_NAME: NVIDIA CUDA
- CL_PLATFORM_VENDOR: NVIDIA Corporation
Number of devices: 1
device: 0
- CL_DEVICE_TYPE: 4
- CL_DEVICE_NAME: NVIDIA GeForce RTX 3090
- CL_DEVICE_MAX_WORK_GROUP_SIZE: 1024
- CL_DEVICE_GLOBAL_MEM_SIZE: 25446907904
- CL_DEVICE_LOCAL_MEM_SIZE: 49152
- CL_DEVICE_MAX_MEM_ALLOC_SIZE: 6361726976
...
*/

  cl_uint num_platforms;
  CHECK_OPENCL(clGetPlatformIDs(0, NULL, &num_platforms));
  cl_platform_id *platforms;
  platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
  CHECK_OPENCL(clGetPlatformIDs(num_platforms, platforms, NULL));
  printf("Number of platforms: %d\n", num_platforms);

  for (int i = 0; i < num_platforms; i++)
  {
    printf("Platform: %d\n", i);
    size_t nameSize;
    char *name;
    CHECK_OPENCL(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &nameSize));
    name = (char *)malloc(nameSize);
    CHECK_OPENCL(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, nameSize, name, NULL));
    printf("_ CL_PLATFORM_NAME: %s\n", name);
    free(name);

    CHECK_OPENCL(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 0, NULL, &nameSize));
    name = (char *)malloc(nameSize);
    CHECK_OPENCL(clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, nameSize, name, NULL));
    printf("_ CL_PLATFORM_VENDOR: %s\n", name);
    free(name);
  }

  return 0;
}
