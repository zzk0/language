#include <chrono>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>
#include <map>
#include <ostream>
#include <string>
#include <vector>

#include "cuda_handle.h"
#include "matmul.h"
#include "smart_pointer.h"
#include "util.h"

#include "tabulate/table.hpp"

using MatmulFunc =
    std::function<void(float *, float *, float *, int, int, int)>;
const std::map<std::string, MatmulFunc> matmul_algorithms{
    // {"cublas", CublasMatmul},
    {"naive", NaiveMatmul},
    {"block", BlockMatmul},
    {"block_stride", BlockWithStrideMatmul},
};

void AddRow(tabulate::Table& table, const std::string& name, int m, int n, int k, bool flag, double duration) {
  table.add_row({
      name, std::to_string(m) + " " + std::to_string(n) +  " " + std::to_string(k),
         flag ? "true" : "false",
         std::to_string(duration) + "ms",
         std::to_string(2.0 * m * n * k / (duration * 1e-3) * 1e-9)
  });
}

void MatmulBenchmark(int m = 3, int n = 4, int k = 5, int times = 100) {
  printf("------------------- bechmark %d times for shape m = %d, n = %d, k = "
         "%d -------------------\n",
         times, m, n, k);
  auto timer = Timer<std::chrono::milliseconds>{};

  // 1. prepare the input data
  timer.Start();
  std::vector<float> A = NewMatrix(m, k);
  std::vector<float> B = NewMatrix(k, n);
  FillRandomNumber(A, m * k);
  FillRandomNumber(B, k * n);
  timer.End();
  printf("data creation time cost: %d milliseconds\n", timer.GetDuration());

  // 2. move the data to device
  timer.Start();
  DeviceUniquePointer<float> A_ptr(A.data(), A.size());
  DeviceUniquePointer<float> B_ptr(B.data(), B.size());
  timer.End();
  printf("data movement time cost: %d milliseconds\n", timer.GetDuration());

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

  // 4.1 prepare the tabulate::Table
  tabulate::Table table;
  table.add_row({"name", "size", "correctness", "latency", "GFLOPs"});
  AddRow(table, "cublas", m, n, k, true, timer.GetDuration() / 100.0);

  // 5. benchmark the custom matmul
  for (auto iter = matmul_algorithms.begin(); iter != matmul_algorithms.end();
       ++iter) {
    const std::string &name = iter->first;
    const auto &func = iter->second;

    // prepare data
    std::vector<float> D = NewMatrix(m, n);
    DeviceUniquePointer<float> D_ptr(D.data(), D.size());

    // correstness check
    func(A_ptr.Get(), B_ptr.Get(), D_ptr.Get(), m, n, k);
    bool flag = EqualCheckCUDA(truth_ptr.Get(), D_ptr.Get(), m * n, true);

    // benchmark
    timer.Start();
    for (int i = 0; i < times; ++i) {
      func(A_ptr.Get(), B_ptr.Get(), D_ptr.Get(), m, n, k);
    }
    timer.End();
    AddRow(table, name, m, n, k, true, timer.GetDuration() / 100.0);
  }

  std::cout << table << std::endl;
}

int main() {
  // MatmulBenchmark(3, 4, 5, 0);
  // MatmulBenchmark(5, 8, 9, 0);
  // MatmulBenchmark(10, 20, 30, 0);
  // MatmulBenchmark(4, 4, 4, 0);
  // MatmulBenchmark(16, 16, 16, 0);
  // MatmulBenchmark(64, 64, 64, 0);
  // MatmulBenchmark(256, 256, 256, 0);
  // MatmulBenchmark(1024, 1024, 1024, 0);
  // MatmulBenchmark(2048, 2048, 2048, 0);

  for (int size = 256; size <= 16384; size += 256) {
    MatmulBenchmark(size, size, size);
  }
  return 0;
}
