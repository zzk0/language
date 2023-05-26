#include "util.h"
#include <chrono>
#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <vector>

#include <cublas_v2.h>
#include <smart_pointer.h>

std::vector<float> NewMatrix(int nx, int ny) {
  return std::vector<float>(nx * ny, 0);
}

void InitMatrix(std::vector<float> &matrix, int nx, int ny) {
  FillRandomNumber<float>(matrix, nx * ny);
}

void PrintMatrix(const std::vector<float> &matrix, int nx, int ny) {
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      std::cout << matrix[i * ny + j] << " ";
    }
    std::cout << std::endl;
  }
}

__global__ void PrintMatrixOnDevice(float *matrix, int nx, int ny) {
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      printf("%f ", matrix[i * ny + j]);
    }
    printf("\n");
  }
}

/**
采用列优先进行实现，与 cuBLAS 的默认行为保持一致。

列优先，(i, j) 对应位置为 i + j * ldx (ldx = 行的个数)
A: (m, k)
B: (k, n)
C: (m, n)

(1024, 1024) * (1024, 1024) -> (1024, 1024)
<<<(128, 256), (8, 4)>>>  每个 block 32 个线程同时执行，启动 1024/blockDim.x, 1024/blockDim.y 个 block

block, grid 的设置：https://zhuanlan.zhihu.com/p/442304996
*/
__global__ void NaiveMatmul(float *A, float *B, float *C, int m, int n, int k) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  if (tx < m && ty < n) {
    float res = 0.0f;
    for (int i = 0; i < k; ++i) {
      // A(tx, i) * B(i, ty) = C(tx, ty)
      res += A[tx + i * m] * B[i + ty * k];
    }
    C[tx + ty * m] = res;
  }
}

void TestCase0() {
  constexpr int nx = 3;
  constexpr int ny = 5;
  std::vector<float> matrix = NewMatrix(nx, ny);
  InitMatrix(matrix, nx, ny);
  std::cout << "Host:" << std::endl;
  PrintMatrix(matrix, nx, ny);

  DeviceUniquePointer<float> ptr(matrix.data(), matrix.size());
  std::cout << "Device:" << std::endl;
  PrintMatrixOnDevice<<<1, 1>>>(ptr.Get(), nx, ny);
}

void TestCase1() {
  constexpr float alpha = 1.0f, beta = 0.0f;
  constexpr int m = 3;
  constexpr int n = 4;
  constexpr int k = 5;
  std::vector<float> A = NewMatrix(k, m);
  std::vector<float> B = NewMatrix(n, k);
  std::vector<float> C = NewMatrix(n, m);
  FillRandomNumber(A, k * m);
  FillRandomNumber(B, n * k);
  FillRandomNumber(C, n * m);
  DeviceUniquePointer<float> A_ptr(A.data(), A.size());
  DeviceUniquePointer<float> B_ptr(B.data(), B.size());
  DeviceUniquePointer<float> C_ptr(C.data(), C.size());

  cublasHandle_t handle;
  cublasCreate(&handle);
  
  {
    /*
    lda, ldb, ldc: 对于行优先，填写一行的元素个数；对于列优先，填写一列的元素个数。
    ldx 存在的原因是，这个矩阵可能是一个大矩阵的一个小矩阵，用来计算偏移量。
    以列优先为例，[i, j] 坐标对应的元素 offset = i + ldx * j (i/j 都从 0 开始)
    */
    TimerRAII<std::chrono::microseconds> timer{"cublas sgemm time cost: %d microseconds\n"};
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A_ptr.Get(), m,
                B_ptr.Get(), k, &beta, C_ptr.Get(), m);
    cudaDeviceSynchronize();
  }
  
  std::cout << "A: " << std::endl;
  PrintMatrixOnDevice<<<1, 1>>>(A_ptr.Get(), k, m);
  cudaDeviceSynchronize();
  std::cout << "B: " << std::endl;
  PrintMatrixOnDevice<<<1, 1>>>(B_ptr.Get(), n, k);
  cudaDeviceSynchronize();
  std::cout << "C: " << std::endl;
  PrintMatrixOnDevice<<<1, 1>>>(C_ptr.Get(), n, m);
  cudaDeviceSynchronize();
  cublasDestroy(handle);
}

void CorrectnessCheck(int m=3, int n=4, int k=5) {
  constexpr float alpha = 1.0f, beta = 0.0f;
  auto timer = Timer<std::chrono::microseconds>{};
  
  timer.Start();
  std::vector<float> A = NewMatrix(k, m);
  std::vector<float> B = NewMatrix(n, k);
  FillRandomNumber(A, k * m);
  FillRandomNumber(B, n * k);
  timer.End();
  printf("data creation time cost: %d microseconds\n", timer.GetDuration());
  
  timer.Start();
  DeviceUniquePointer<float> A_ptr(A.data(), A.size());
  DeviceUniquePointer<float> B_ptr(B.data(), B.size());
  timer.End();
  printf("data movement time cost: %d microseconds\n", timer.GetDuration());

  std::vector<float> C = NewMatrix(n, m);
  DeviceUniquePointer<float> C_ptr(C.data(), C.size());
  cublasHandle_t handle;
  cublasCreate(&handle);
  {
    TimerRAII<std::chrono::microseconds> timer{"cublas sgemm time cost: %d microseconds\n"};
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A_ptr.Get(), m,
                B_ptr.Get(), k, &beta, C_ptr.Get(), m);
    cudaDeviceSynchronize();
  }
  cublasDestroy(handle);

  std::vector<float> D = NewMatrix(n, m);
  DeviceUniquePointer<float> D_ptr(C.data(), C.size());
  {
    TimerRAII<std::chrono::microseconds> timer{"naive sgemm time cost: %d microseconds\n"};
    dim3 grid(128, 256);
    dim3 block(8, 4);
    NaiveMatmul<<<grid, block>>>(A_ptr.Get(), B_ptr.Get(), D_ptr.Get(), m, n, k);
    cudaDeviceSynchronize();
  }
  cudaDeviceSynchronize();

  bool equal = EqualCheckCUDA(C_ptr.Get(), D_ptr.Get(), m * n);
  if (equal) {
    printf("equal, status: %d\n", equal);
  }
  else {
    printf("not equal, status: %d\n", equal);
  }
}

void MatmulBenchmark(int size=1024) {
  printf("---------------------- size = %d ----------------------\n", size);
  constexpr float alpha = 1.0f, beta = 0.0f;
  int m = size;
  int n = size;
  int k = size;

  auto timer = Timer<std::chrono::microseconds>{};
  
  timer.Start();
  std::vector<float> A = NewMatrix(k, m);
  std::vector<float> B = NewMatrix(n, k);
  FillRandomNumber(A, k * m);
  FillRandomNumber(B, n * k);
  timer.End();
  printf("data creation time cost: %d microseconds\n", timer.GetDuration());
  
  timer.Start();
  DeviceUniquePointer<float> A_ptr(A.data(), A.size());
  DeviceUniquePointer<float> B_ptr(B.data(), B.size());
  timer.End();
  printf("data movement time cost: %d microseconds\n", timer.GetDuration());

  std::vector<float> C = NewMatrix(n, m);
  DeviceUniquePointer<float> C_ptr(C.data(), C.size());
  cublasHandle_t handle;
  cublasCreate(&handle);
  {
    TimerRAII<std::chrono::microseconds> timer{"cublas sgemm time cost: %d microseconds\n"};
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A_ptr.Get(), m,
                B_ptr.Get(), k, &beta, C_ptr.Get(), m);
    cudaDeviceSynchronize();
  }
  cublasDestroy(handle);

  {
    TimerRAII<std::chrono::microseconds> timer{"naive sgemm time cost: %d microseconds\n"};
    const auto& t = GetGridAndBlock(m, n);
    const auto& grid = std::get<0>(t);
    const auto& block = std::get<1>(t);
    NaiveMatmul<<<grid, block>>>(A_ptr.Get(), B_ptr.Get(), C_ptr.Get(), m, n, k);
    cudaDeviceSynchronize();
  }
  cudaDeviceSynchronize();
}


int main() {
  CorrectnessCheck(3, 4, 5);
  CorrectnessCheck(10, 20, 30);
  CorrectnessCheck(5, 8, 9);
  CorrectnessCheck(1024, 1024, 1024);

  // MatmulBenchmark(1024);
  // MatmulBenchmark(2048);
  // MatmulBenchmark(4096);
  // MatmulBenchmark(8192);
  // MatmulBenchmark(16384);
  return 0;
}
