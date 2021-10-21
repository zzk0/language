//
// Created by zzk on 2021/10/20.
//

#include "iostream"
#include "stdio.h"

__global__ void hello_world() {
//  std::cout << "Hello Cuda" << std::endl;
  printf("Hello CUDA\n");
}

int main() {
  std::cout << "Hello CPU" << std::endl;
  hello_world<<<1, 10>>>();
  cudaDeviceReset();
  return 0;
}
