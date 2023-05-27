#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdio.h>
#include <vector>

#include "smart_pointer.h"
#include "util.h"

void AddMatricesOnHost(float *a, float *b, float *c, int nx, int ny) {
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      c[i * ny + j] = a[i * ny + j] + b[i * ny + j];
    }
  }
}

/**
1d grid && 1d block
*/
__global__ void AddMatrices(float *a, float *b, float *c, int nx, int ny) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  c[i] = a[i] + b[i];
}
 
/**
2d grid && 2d block
*/
__global__ void AddMatrices22(float *a, float *b, float *c, int nx, int ny) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < nx && j < ny) {
    unsigned int id = i * ny + j;
    c[id] = a[id] + b[id];
  }
}

/**
1d grid && 1d block for 2x2 configuration
*/
__global__ void AddMatrices11(const float *a, const float *b, float *c, int nx, int ny) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nx) {
    for (int j = 0; j < ny; ++j) {
      unsigned int id = i * ny + j;
      c[id] = a[id] + b[id];
    }
  }
}

/**
2d grid && 1d block for 2x2 configuration
*/
__global__ void AddMatrices21(float *a, float *b, float *c, int nx, int ny) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y;  // blockDim.y == 1
  if (i < nx && j < ny) {
    unsigned int id = i * ny + j;
    c[id] = a[id] + b[id];
  }
}

bool CompareMatrices(float *a, float *b, int nx, int ny) {
  float epsilon = 1e-8;
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      if (std::abs(a[i * ny + j] - b[i * ny + j]) > epsilon) {
        std::cout << i << " " << j << " " << a[i * ny + j] << " " << b[i * ny + j] << std::endl;
        return false;
      }
    }
  }
  return true;
}

void TestCase0() {
  constexpr int nx = 3;
  constexpr int ny = 5;
  std::vector<float> matrix = NewMatrix(nx, ny);
  InitMatrix(matrix, nx, ny);
  PrintMatrix(matrix, nx, ny);

  DeviceUniquePointer<float> ptr(matrix.data(), matrix.size());
  PrintMatrixOnDevice(ptr.Get(), nx, ny);

  std::cout << CompareMatrices(matrix.data(), ptr.GetHostPtr().get(), nx, ny)
            << std::endl;
}

void TestCase1() {  
  constexpr int nx = 3;
  constexpr int ny = 5;

  std::vector<float> ha = NewMatrix(nx, ny);
  std::vector<float> hb = NewMatrix(nx, ny);
  std::vector<float> hc = NewMatrix(nx, ny);
  InitMatrix(ha, nx, ny);
  InitMatrix(hb, nx, ny);
  DeviceUniquePointer<float> da(ha.data(), ha.size());
  DeviceUniquePointer<float> db(hb.data(), hb.size());
  DeviceUniquePointer<float> dc(hc.data(), hc.size());

  AddMatricesOnHost(ha.data(), hb.data(), hc.data(), nx, ny);
  AddMatrices<<<nx, ny>>>(da.Get(), db.Get(), dc.Get(), nx, ny);
  std::cout << CompareMatrices(hc.data(), dc.GetHostPtr().get(), nx, ny)
            << std::endl;

}

void TestCase2() {
  constexpr int nx = 1 << 14;
  constexpr int ny = 1 << 13;

  std::vector<float> ha = NewMatrix(nx, ny);
  std::vector<float> hb = NewMatrix(nx, ny);
  std::vector<float> hc = NewMatrix(nx, ny);
  InitMatrix(ha, nx, ny);
  InitMatrix(hb, nx, ny);
  DeviceUniquePointer<float> da(ha.data(), ha.size());
  DeviceUniquePointer<float> db(hb.data(), hb.size());

  { 
    TimerRAII<std::chrono::microseconds> timer("CPU time cost: %d us\n");
    AddMatricesOnHost(ha.data(), hb.data(), hc.data(), nx, ny);
  }

  auto matrix_sum = [&](int block_x, int block_y, int grid_x, int grid_y) {
    DeviceUniquePointer<float> dc(sizeof(float) * hc.size());
    dim3 block(block_x, block_y);
    dim3 grid(grid_x, grid_y);
    char buf[1024];
    sprintf(buf, "<<<(%d, %d), (%d, %d)>>>", grid.x, grid.y, block.x, block.y);

    {
      TimerRAII<std::chrono::microseconds> timer(std::string(buf) + " time cost: %d us\n");
      AddMatrices22<<<grid, block>>>(da.Get(), db.Get(), dc.Get(), nx, ny);
      cudaDeviceSynchronize();
    }

    bool res = CompareMatrices(hc.data(), dc.GetHostPtr().get(), nx, ny);
    if (res) {
      std::cout << "Equal" << std::endl;
    }
    else {
      std::cout << "Not Equal" << std::endl;
    }
  };

  matrix_sum(32, 32, (nx + 32 - 1) / 32, (ny + 32 - 1) / 32);
  matrix_sum(32, 16, (nx + 32 - 1) / 32, (ny + 16 - 1) / 16);
  matrix_sum(16, 16, (nx + 16 - 1) / 16, (ny + 16 - 1) / 16);

  matrix_sum(128, 1, (nx + 32 - 1) / 32, 1);
  matrix_sum(128, 1, (nx + 32 - 1) / 32, ny);
}

void TestCase3() {
  constexpr int nx = 3;
  constexpr int ny = 5;
  std::vector<float> ha = NewMatrix(nx, ny);
  std::vector<float> hb = NewMatrix(nx, ny);
  std::vector<float> hc = NewMatrix(nx, ny);
  InitMatrix(ha, nx, ny);
  InitMatrix(hb, nx, ny);
  AddMatricesOnHost(ha.data(), hb.data(), hc.data(), nx, ny);
  PrintMatrix(ha, nx, ny);
  PrintMatrix(hb, nx, ny);
  PrintMatrix(hc, nx, ny);
}

int main() {
  TestCase2();

  cudaDeviceSynchronize();
  return 0;
}
