#include <chrono>
#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <unordered_map>
#include <vector>
#include <cublas_v2.h>

#include "smart_pointer.h"
#include "cuda_handle.h"
#include "util.h"
#include "matmul.h"

using MatmulFunc = std::function<void(float*, float*, float*, int, int, int)>;
const std::unordered_map<std::string, MatmulFunc> matmul_algorithms {
  // {"cublas", CublasMatmul},
  {"naive", NaiveMatmul}
};

void MatmulBenchmark(int m=3, int n=4, int k=5, int times=100, int correctness_check=3) {
  printf("------------------- bechmark %d times for shape m = %d, n = %d, k = %d -------------------\n", times, m, n, k);
  auto timer = Timer<std::chrono::microseconds>{};
  
  // 1. prepare the input data
  timer.Start();
  std::vector<float> A = NewMatrix(m, k);
  std::vector<float> B = NewMatrix(k, n);
  FillRandomNumber(A, m * k);
  FillRandomNumber(B, k * n);
  timer.End();
  printf("data creation time cost: %d microseconds\n", timer.GetDuration());
  
  // 2. move the data to device
  timer.Start();
  DeviceUniquePointer<float> A_ptr(A.data(), A.size());
  DeviceUniquePointer<float> B_ptr(B.data(), B.size());
  timer.End();
  printf("data movement time cost: %d microseconds\n", timer.GetDuration());

  // 3. prepare the true data
  std::vector<float> truth = NewMatrix(m, n);
  DeviceUniquePointer<float> truth_ptr(truth.data(), truth.size());
  CublasMatmul(A_ptr.Get(), B_ptr.Get(), truth_ptr.Get(), m, n, k);

  // 4. benchmark the cublas
  timer.Start();
  for (int i = 0; i < times; ++i) {
    CublasMatmul(A_ptr.Get(), B_ptr.Get(), truth_ptr.Get(), m, n, k);
  }
  timer.End();
  printf("%d cublas matmul time cost: %d microseconds\n", times, timer.GetDuration());
  
  // 5. benchmark the custom matmul
  for (auto iter = matmul_algorithms.begin(); iter != matmul_algorithms.end(); ++iter) {
    const std::string& name = iter->first;
    const auto& func = iter->second;

    // prepare data
    std::vector<float> D = NewMatrix(m, n);
    DeviceUniquePointer<float> D_ptr(D.data(), D.size());

    // correstness check
    for (int i = 0; i < correctness_check; ++i) {
      func(A_ptr.Get(), B_ptr.Get(), D_ptr.Get(), m, n, k);
      EqualCheckCUDA(truth_ptr.Get(), D_ptr.Get(), m * n, true);
    }
  
    // benchmark
    timer.Start();
    for (int i = 0; i < times; ++i) {
      func(A_ptr.Get(), B_ptr.Get(), D_ptr.Get(), m, n, k);
    }
    timer.End();
    printf("%d %s matmul time cost: %d microseconds\n", times, name.c_str(), timer.GetDuration());
  }
}


int main() {
  MatmulBenchmark(3, 4, 5);
  MatmulBenchmark(10, 20, 30);
  MatmulBenchmark(5, 8, 9);
  MatmulBenchmark(1024, 1024, 1024);
  MatmulBenchmark(2048, 2048, 2048);
  MatmulBenchmark(4096, 4096, 4096);
  MatmulBenchmark(8192, 8192, 8192);
  MatmulBenchmark(16384, 16384, 16384);
  return 0;
}
