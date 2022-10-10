#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdio.h>
#include <vector>

#include "smart_pointer.h"
#include "util.h"

__global__ void MathKernel1(float *c) {
  float a = 0.0f;
  float b = 0.0f;

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid % 2 == 0) {
    a = 100.0f;
  }
  else {
    b = 200.0f;
  }
  c[tid] = a + b;
}

__global__ void MathKernel2(float *c) {
  float a = 0.0f;
  float b = 0.0f;

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if ((tid / warpSize) % 2 == 0) {
    a = 100.0f;
  }
  else {
    b = 200.0f;
  }
  c[tid] = a + b;
}

void TestCase0() {
  auto PrintArray = [](float *arr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
  };

  constexpr size_t size = 1024;
  DeviceUniquePointer<float> ptr(size * sizeof(float));
  {
    TimerRAII<std::chrono::microseconds> timer("warmup cost: %d us\n");
    MathKernel1<<<1, size>>>(ptr.Get());
  }
  // PrintArray(ptr.GetHostPtr().get(), size);
  
  {
    TimerRAII<std::chrono::microseconds> timer("MathKernel1: %d us\n");
    MathKernel1<<<1, size>>>(ptr.Get());
  }
  // PrintArray(ptr.GetHostPtr().get(), size);

  {
    TimerRAII<std::chrono::microseconds> timer("MathKernl2: %d us\n");
    MathKernel2<<<1, size>>>(ptr.Get());
  }
  // PrintArray(ptr.GetHostPtr().get(), size);
}


int main() {
  TestCase0();

  cudaDeviceSynchronize();
  return 0;
}
