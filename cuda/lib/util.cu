#include "util.h"
#include <cuda_device_runtime_api.h>
#include <driver_types.h>

namespace {

/**
用于比较两个矩阵是否相同。
列优先，(i, j) 对应位置为 i + j * ldx (ldx = 行的个数)
*/
__global__ void EqualCheckCUDAimpl(float *A, float *B, int numel, int *status) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tx < numel) {
    if (abs(A[tx] - B[tx]) > 1e-3) {
      printf("not equal: %f %f\n", A[tx], B[tx]);
      *status += 1;
    }
  }
}

}

bool EqualCheckCUDA(float *dev_a, float *dev_b, int numel, bool verbose) {
  const auto& t = GetGridAndBlock(numel);
  const auto& grid = std::get<0>(t);
  const auto& block = std::get<1>(t);
  int status = 0;
  int *dev_status;
  cudaMalloc(&dev_status, sizeof(int));
  cudaMemcpy(dev_status, &status, sizeof(int), cudaMemcpyHostToDevice);
  EqualCheckCUDAimpl<<<grid, block>>>(dev_a, dev_b, numel, dev_status);
  cudaMemcpy(&status, dev_status, sizeof(int), cudaMemcpyDeviceToHost);
  return (status == 0);
}
