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
  } else {
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
  } else {
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

template <typename T> bool CompareResult(const T a, const T b) {
  float epsilon = 0.1;
  if (std::abs(float(a) - float(b)) > epsilon) {
    std::cout << a << " " << b << " ";
    return false;
  }
  return true;
}

template <typename T> void CpuReduceSum(T *idata, T *odata, unsigned int size) {
  odata[0] = 0.0f;
  for (unsigned int i = 0; i < size; ++i) {
    odata[0] += idata[i];
  }
}

typedef void (*ReduceFunctor)(int *, int *, unsigned int);

/**
reduce neighbor
*/
template <typename T>
__global__ void GpuReduceSum0(T *idata, T *odata, unsigned int size) {
  if (blockIdx.x * blockDim.x + threadIdx.x >= size) {
    return;
  }
  unsigned int tid = threadIdx.x;
  T *pidata = idata + (blockIdx.x * blockDim.x);
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

/**
reduce neighbor + less divergence
*/
template <typename T>
__global__ void GpuReduceSum1(T *idata, T *odata, unsigned int size) {
  if (blockIdx.x * blockDim.x + threadIdx.x >= size) {
    return;
  }
  unsigned int tid = threadIdx.x;
  T *pidata = idata + (blockIdx.x * blockDim.x);
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    unsigned int index = 2 * stride * tid;
    if (index < blockDim.x) {
      pidata[index] += pidata[index + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    odata[blockIdx.x] = pidata[0];
  }
}

/**
reduce interleave
*/
template <typename T>
__global__ void GpuReduceSum2(T *idata, T *odata, unsigned int size) {
  if (blockIdx.x * blockDim.x + threadIdx.x >= size) {
    return;
  }
  unsigned int tid = threadIdx.x;
  T *pidata = idata + blockIdx.x * blockDim.x;
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      pidata[tid] += pidata[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    odata[blockIdx.x] = pidata[0];
  }
}

/**
reduce interleave + loop unrolling x 2
ps. 有一次运行 nvprof ./cuda/reduce_sum/reduce_sum 发生了数据不一致的情况
*/
template <typename T>
__global__ void GpuReduceSum3(T *idata, T *odata, unsigned int size) {
  unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  unsigned int tid = threadIdx.x;
  T *pidata = idata + blockIdx.x * blockDim.x * 2;
  if (idx + blockDim.x < size) {
    pidata[idx] += pidata[idx + blockDim.x];
  }
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
    if (tid < stride) {
      pidata[tid] += pidata[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    odata[blockIdx.x] = pidata[0];
  }
}

/**
reduce interleave + loop unrolling x 8
*/
template <typename T>
__global__ void GpuReduceSum4(T *idata, T *odata, unsigned int size) {
  unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
  unsigned int tid = threadIdx.x;
  T *pidata = idata + blockIdx.x * blockDim.x * 8;
  if (idx + blockDim.x * 7 < size) {
    int a1 = pidata[idx + blockDim.x];
    int a2 = pidata[idx + blockDim.x * 2];
    int a3 = pidata[idx + blockDim.x * 3];
    int a4 = pidata[idx + blockDim.x * 4];
    int a5 = pidata[idx + blockDim.x * 5];
    int a6 = pidata[idx + blockDim.x * 6];
    int a7 = pidata[idx + blockDim.x * 7];
    pidata[idx] += (a1 + a2 + a3 + a4 + a5 + a6 + a7);
  }
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride >= 1; stride >>= 1) {
    if (tid < stride) {
      pidata[tid] += pidata[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    odata[blockIdx.x] = pidata[0];
  }
}

/**
reduce interleave + loop unrolling x 8 + warp unrolling
*/
template <typename T>
__global__ void GpuReduceSum5(T *idata, T *odata, unsigned int size) {
  unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
  unsigned int tid = threadIdx.x;
  T *pidata = idata + blockIdx.x * blockDim.x * 8;
  if (idx + blockDim.x * 7 < size) {
    int a1 = pidata[idx + blockDim.x];
    int a2 = pidata[idx + blockDim.x * 2];
    int a3 = pidata[idx + blockDim.x * 3];
    int a4 = pidata[idx + blockDim.x * 4];
    int a5 = pidata[idx + blockDim.x * 5];
    int a6 = pidata[idx + blockDim.x * 6];
    int a7 = pidata[idx + blockDim.x * 7];
    pidata[idx] += (a1 + a2 + a3 + a4 + a5 + a6 + a7);
  }
  __syncthreads();

  for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride) {
      pidata[tid] += pidata[tid + stride];
    }
    __syncthreads();
  }

  if (tid < 32) {
    volatile int *vmem = pidata;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
  }

  if (tid == 0) {
    odata[blockIdx.x] = pidata[0];
  }
}

/**
reduce interleave + loop unrolling x 8 + warp unrolling + complete unroll
*/
template <typename T>
__global__ void GpuReduceSum6(T *idata, T *odata, unsigned int size) {
  unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
  unsigned int tid = threadIdx.x;
  T *pidata = idata + blockIdx.x * blockDim.x * 8;
  if (idx + blockDim.x * 7 < size) {
    int a1 = pidata[idx + blockDim.x];
    int a2 = pidata[idx + blockDim.x * 2];
    int a3 = pidata[idx + blockDim.x * 3];
    int a4 = pidata[idx + blockDim.x * 4];
    int a5 = pidata[idx + blockDim.x * 5];
    int a6 = pidata[idx + blockDim.x * 6];
    int a7 = pidata[idx + blockDim.x * 7];
    pidata[idx] += (a1 + a2 + a3 + a4 + a5 + a6 + a7);
  }
  __syncthreads();

  if (blockDim.x >= 1024 && tid < 512) {
    pidata[tid] += pidata[tid + 512];
  }
  __syncthreads();

  if (blockDim.x >= 512 && tid < 256) {
    pidata[tid] += pidata[tid + 256];
  }
  __syncthreads();

  if (blockDim.x >= 256 && tid < 128) {
    pidata[tid] += pidata[tid + 128];
  }
  __syncthreads();

  if (blockDim.x >= 128 && tid < 64) {
    pidata[tid] += pidata[tid + 64];
  }
  __syncthreads();

  if (tid < 32) {
    volatile int *vmem = pidata;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
  }

  if (tid == 0) {
    odata[blockIdx.x] = pidata[0];
  }
}

/**
reduce interleave + loop unrolling x 8 + warp unrolling + complete unroll +
template
*/
template <typename T, unsigned int t_block_size>
__global__ void GpuReduceSum7(T *idata, T *odata, unsigned int size) {
  unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;
  unsigned int tid = threadIdx.x;
  T *pidata = idata + blockIdx.x * blockDim.x * 8;
  if (idx + blockDim.x * 7 < size) {
    int a1 = pidata[idx + blockDim.x];
    int a2 = pidata[idx + blockDim.x * 2];
    int a3 = pidata[idx + blockDim.x * 3];
    int a4 = pidata[idx + blockDim.x * 4];
    int a5 = pidata[idx + blockDim.x * 5];
    int a6 = pidata[idx + blockDim.x * 6];
    int a7 = pidata[idx + blockDim.x * 7];
    pidata[idx] += (a1 + a2 + a3 + a4 + a5 + a6 + a7);
  }
  __syncthreads();

  if (t_block_size >= 1024 && tid < 512) {
    pidata[tid] += pidata[tid + 512];
  }
  __syncthreads();

  if (t_block_size >= 512 && tid < 256) {
    pidata[tid] += pidata[tid + 256];
  }
  __syncthreads();

  if (t_block_size >= 256 && tid < 128) {
    pidata[tid] += pidata[tid + 128];
  }
  __syncthreads();

  if (t_block_size >= 128 && tid < 64) {
    pidata[tid] += pidata[tid + 64];
  }
  __syncthreads();

  if (tid < 32) {
    volatile int *vmem = pidata;
    vmem[tid] += vmem[tid + 32];
    vmem[tid] += vmem[tid + 16];
    vmem[tid] += vmem[tid + 8];
    vmem[tid] += vmem[tid + 4];
    vmem[tid] += vmem[tid + 2];
    vmem[tid] += vmem[tid + 1];
  }

  if (tid == 0) {
    odata[blockIdx.x] = pidata[0];
  }
}

// template<typename T>
// __global__ void ReduceSumRecursive(T *idata, T *odata, unsigned int size) {
//   unsigned int tid = threadIdx.x;
//   T *pidata = idata + blockIdx.x * blockDim.x;
//   T *podata = &odata[blockIdx.x];

//   if (size == 2 && tid == 0) {
//     odata[blockIdx.x] = idata[0] + idata[1];
//     return;
//   }

//   unsigned int stride = size >> 1;
//   if (stride > 1 && tid < stride) {
//     pidata[tid] = pidata[tid + stride];
//   }

//   __syncthreads();

//   if (tid == 0) {
//     ReduceSumRecursive<<<1, stride>>>(pidata, podata, stride);
//     cudaDeviceSynchronize();
//   }
//   __syncthreads();
// }

void TestCase1() {
  constexpr unsigned int element_cnt = 1 << 24;
  std::vector<int> idata(element_cnt, 0);
  int odata = 0;
  FillRandomNumber<int>(idata, element_cnt, -10, 10);

  {
    TimerRAII<std::chrono::microseconds> timer(
        "CPU reduce sum time cost: %d us\n");
    CpuReduceSum(idata.data(), &odata, element_cnt);
  }

  {
    DeviceUniquePointer<int> iptr(idata.data(), idata.size());
    DeviceUniquePointer<int> optr(&odata, 1);
    {
      TimerRAII<std::chrono::microseconds> timer(
          "GPU warmup time cost: %d us\n");
      GpuReduceSum0<<<1, 1>>>(iptr.Get(), optr.Get(), element_cnt);
      cudaDeviceSynchronize();
    }
  }

  const auto &ReduceSum = [&](const ReduceFunctor &functor,
                              const std::string &functor_name,
                              unsigned int block_size, unsigned int grid_size) {
    DeviceUniquePointer<int> iptr(idata.data(), idata.size());
    DeviceUniquePointer<int> optr(sizeof(int) * grid_size);
    int gpu_sum = 0;
    {
      TimerRAII<std::chrono::microseconds> timer("time cost: %d us \t" +
                                                 functor_name + "\n");
      functor<<<{grid_size, 1}, {block_size, 1}>>>(iptr.Get(), optr.Get(),
                                                   element_cnt);
      cudaDeviceSynchronize();
    }
    const auto &host_ptr = optr.GetHostPtr(); // keep this unique_ptr, otherwise
                                              // gpu_odata will be released
    int *gpu_odata = host_ptr.get();
    for (unsigned int i = 0; i < grid_size; ++i) {
      gpu_sum += gpu_odata[i];
    }
    bool ans = CompareResult(odata, gpu_sum);
    if (!ans) {
      std::cout << "\033[31mNot Equal\033[0m" << std::endl;
      exit(1);
    }
    // if (ans) {
    //   std::cout << "\033[32mEqual\033[0m" << std::endl;
    // } else {
    //   std::cout << "\033[31mNot Equal\033[0m" << std::endl;
    // }
  };

  constexpr unsigned int block_size = 512;
  unsigned int grid_size = (element_cnt + block_size - 1) / block_size;
  ReduceSum(GpuReduceSum0, "reduce neighbor ", block_size, grid_size);
  ReduceSum(GpuReduceSum1, "reduce neighbor + less divergence ", block_size,
            grid_size);
  ReduceSum(GpuReduceSum2, "reduce interleave ", block_size, grid_size);
  ReduceSum(GpuReduceSum3, "reduce interleave + loop unrolling x 2 ",
            block_size, grid_size / 2);
  ReduceSum(GpuReduceSum4, "reduce interleave + loop unrolling x 8 ",
            block_size, grid_size / 8);
  ReduceSum(GpuReduceSum5,
            "reduce interleave + loop unrolling x 8 + warp unrolling ",
            block_size, grid_size / 8);
  ReduceSum(GpuReduceSum6,
            "reduce interleave + loop unrolling x 8 + warp unrolling + "
            "complete unroll ",
            block_size, grid_size / 8);
  ReduceSum(GpuReduceSum7<int, block_size>,
            "reduce interleave + loop unrolling x 8 + warp unrolling + "
            "complete unroll + template ",
            block_size, grid_size / 8);
  // ReduceSum(ReduceSumRecursive<int>,
  //           "reduce sum recursive ",
  //           block_size, grid_size);
}

int main() {
  TestCase1();

  cudaDeviceSynchronize();
  return 0;
}
