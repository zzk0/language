#include "iostream"
#include "stdio.h"

__device__ int devData;
__global__ void counter() {
  printf("%i\n", devData);
  devData += 1;
}

__global__ void hello_world() {
//  std::cout << "Hello Cuda" << std::endl;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  printf("%d %d\n", i, j);
}

int main() {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  std::cout << "Device Count: " << deviceCount << std::endl;

  int x = cudaDeviceEnablePeerAccess(1, 0);
  std::cout << (x == cudaSuccess) << std::endl;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << prop.name << std::endl;
  std::cout << prop.multiProcessorCount << std::endl;
  std::cout << prop.sharedMemPerBlock << std::endl;
  std::cout << prop.maxThreadsPerBlock << std::endl;
  std::cout << prop.maxThreadsPerMultiProcessor << std::endl;
  std::cout << prop.maxGridSize[0] << " " << prop.maxGridSize[1] << " " << prop.maxGridSize[2] << std::endl;

  hello_world<<<1, 5>>>();
  cudaDeviceSynchronize();

  return 0;
}
