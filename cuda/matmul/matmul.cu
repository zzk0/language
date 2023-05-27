#include "util.h"
#include "cuda_handle.h"

namespace matmul {

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
__global__ void NaiveMatmulImpl(float *A, float *B, float *C, int m, int n, int k) {
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

}

void NaiveMatmul(float *A, float *B, float *C, int m, int n, int k) {
  const auto& t = GetGridAndBlock(m, n);
  const auto& grid = std::get<0>(t);
  const auto& block = std::get<1>(t);
  {
    matmul::NaiveMatmulImpl<<<grid, block>>>(A, B, C, m, n, k);
    cudaDeviceSynchronize();
  }
}

void CublasMatmul(float *A, float *B, float *C, int m, int n, int k) {
  CublasHandle& cublas_handle = CublasHandle::GetInstance();
  auto handle = cublas_handle.GetCublasHandle();
  constexpr float alpha = 1.0f, beta = 0.0f;
  {
    /*
    lda, ldb, ldc: 对于行优先，填写一行的元素个数；对于列优先，填写一列的元素个数。
    ldx 存在的原因是，这个矩阵可能是一个大矩阵的一个小矩阵，用来计算偏移量。
    以列优先为例，[i, j] 坐标对应的元素 offset = i + ldx * j (i/j 都从 0 开始)
    */
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m,
                B, k, &beta, C, m);
    cudaDeviceSynchronize();
  }
}
