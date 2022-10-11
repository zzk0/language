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
  auto PrintArray = [](float *arr, unsigned int size) {
    for (unsigned int i = 0; i < size; ++i) {
      std::cout << arr[i] << " ";
    }
    std::cout << std::endl;
  };

  constexpr unsigned int size = 1024;
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

template<typename T>
bool CompareResult(const T a, const T b) {
  float epsilon = 0.1;
  if (std::abs(float(a) - float(b)) > epsilon) {
    std::cout << a << " " << b << " ";
    return false;
  }
  return true;
}

template<typename T>
void CpuReduceSum(T *idata, T *odata, unsigned int size) {
  odata[0] = 0.0f;
  for (unsigned int i = 0; i < size; ++i) {
    odata[0] += idata[i];
  }
}

/**
reduce neighbor
*/
template<typename T>
__global__ void GpuReduceSum0(T *idata, T *odata, unsigned int size) {
  unsigned int tid = threadIdx.x;
  T *pidata = idata + (blockIdx.x * blockDim.x);
  if (blockIdx.x * blockDim.x + threadIdx.x >= size) {
    return;
  }
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    if ((tid % (2 * stride)) == 0) {
      pidata[tid] += pidata[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    odata[blockIdx.x] = pidata[0];
  }
}

void TestCase1() {
  constexpr unsigned int element_cnt = 1 << 24;
  std::vector<int> idata(element_cnt, 0);
  int odata = 0;
  FillRandomNumber<int>(idata, element_cnt, -10, 10);

  {
    TimerRAII<std::chrono::microseconds> timer("CPU reduce sum time cost: %d us\n");
    CpuReduceSum(idata.data(), &odata, element_cnt);
  }

  {
    DeviceUniquePointer<int> iptr(idata.data(), idata.size());
    DeviceUniquePointer<int> optr(&odata, 1);
    {
      TimerRAII<std::chrono::microseconds> timer("GPU warmup time cost: %d us\n");
      GpuReduceSum0<<<1, 1>>>(iptr.Get(), optr.Get(), element_cnt);
      cudaDeviceSynchronize();
    }
  }

  {
    unsigned int block_size = 512;
    unsigned int grid_size = (element_cnt + block_size - 1) / block_size;
    DeviceUniquePointer<int> iptr(idata.data(), idata.size());
    DeviceUniquePointer<int> optr(sizeof(int) * grid_size);
    int gpu_sum = 0.0f;
    {
      TimerRAII<std::chrono::microseconds> timer("GPU reduce sum time cost: %d us ");
      GpuReduceSum0<<<{grid_size, 1}, {block_size, 1}>>>(iptr.Get(), optr.Get(), element_cnt);
      cudaDeviceSynchronize();
    }
    const auto& host_ptr = optr.GetHostPtr();  // keep this unique_ptr, otherwise gpu_odata will be released
    int *gpu_odata = host_ptr.get();
    for (unsigned int i = 0; i < grid_size; ++i) {
      gpu_sum += gpu_odata[i];
    }
    bool ans = CompareResult(odata, gpu_sum);
    if (ans) {
      std::cout << "Equal" << std::endl;
    }
    else {
      std::cout << "Not Equal" << std::endl;
    }
  }
}

int main() {
  TestCase1();

  cudaDeviceSynchronize();
  return 0;
}
