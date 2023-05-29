#include "cuda_handle.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util.h"

namespace matmul {

enum class MatrixOrder { RowMajor, ColMajor };

template <typename type, MatrixOrder order = MatrixOrder::ColMajor>
class Matrix {
public:
  Matrix(type *dev_ptr, int m, int n) : dev_ptr_(dev_ptr), m_(m), n_(n) {
    if (order == MatrixOrder::ColMajor) {
      lda_ = 1;
      ldb_ = m;
    } else {
      lda_ = n;
      ldb_ = 1;
    }
  }
  ~Matrix() {}
  type *operator()(int i, int j) { return &dev_ptr_[i * lda_ + j * ldb_]; }

private:
  type *dev_ptr_;
  int m_;
  int n_;
  int lda_;
  int ldb_;
};

/**
采用列优先进行实现，与 cuBLAS 的默认行为保持一致。

列优先，(i, j) 对应位置为 i + j * ldx (ldx = 行的个数)
行优先，(i, j) 对应位置为 i * ldx + j (ldx = 列的个数)
A: (m, k)
B: (k, n)
C: (m, n)

(1024, 1024) * (1024, 1024) -> (1024, 1024)
<<<(128, 256), (8, 4)>>>  每个 block 32 个线程同时执行，启动 1024/blockDim.x,
1024/blockDim.y 个 block

block, grid 的设置：https://zhuanlan.zhihu.com/p/442304996
*/
__global__ void NaiveMatmulImpl(float *A, float *B, float *C, int m, int n,
                                int k) {
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

/**
使用二维的共享内存数组，速度反而变慢了？？
可能的原因是：
*/
// __global__ void BlockMatmulImpl(float *A, float *B, float *C, int m, int n, int k) {
//   int tx = blockIdx.x * blockDim.x + threadIdx.x;
//   int ty = blockIdx.y * blockDim.y + threadIdx.y;
//   constexpr int block_size = 16;
//   int matrix_size = m;
//   int block_num = matrix_size / block_size;
  
//   float sum = 0.0f;
//   for (int b = 0; b < block_num; ++b) {
//     // read elements to shared memory
//     __shared__ float sa[block_size][block_size];
//     __shared__ float sb[block_size][block_size];
//     sa[threadIdx.x][threadIdx.y] = A[tx + (ty % block_size + b * block_size) * m];
//     sb[threadIdx.x][threadIdx.y] = B[(tx % block_size + b * block_size) + ty * k];
//     __syncthreads();

//     // compute
//     #pragma unroll
//     for (int k = 0; k < block_size; ++k) {
//       sum += sa[threadIdx.x][k] * sb[k][threadIdx.y];
//     }
//     __syncthreads();
//   }
//   C[tx + ty * m] = sum;
// }


/**
这个 kernel 要求 m == n == k, blockDim.x == blockDim.y
对于矩阵 A B C，其列优先的 ldx 分别是 m, k, m
*/
__global__ void BlockMatmulImpl(float *A, float *B, float *C, int m, int n, int k) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  constexpr int block_size = 16;
  int matrix_size = m;
  int block_num = matrix_size / block_size;
  
  float sum = 0.0f;
  for (int b = 0; b < block_num; ++b) {
    // read elements to shared memory
    __shared__ float sa[block_size * block_size];
    __shared__ float sb[block_size * block_size];
    sa[threadIdx.x + threadIdx.y * block_size] = A[tx + (ty % block_size + b * block_size) * m];
    sb[threadIdx.x + threadIdx.y * block_size] = B[(tx % block_size + b * block_size) + ty * k];
    __syncthreads();

    // compute
    #pragma unroll
    for (int k = 0; k < block_size; ++k) {
      sum += sa[threadIdx.x + k * block_size] * sb[k + threadIdx.y * block_size];
    }
    __syncthreads();
  }
  C[tx + ty * m] = sum;
}

/**
一个线程计算 stride * stride 个元素；因此启动的 block 数量相应减少为 block / stride / stride
要求矩阵的元素数量大于 16 * stride(2) = 32
*/
template<int stride>
__global__ void BlockWithStrideMatmulImpl(float *A, float *B, float *C, int m, int n, int k) {
  constexpr int block_size = 16 * stride;
  int matrix_size = m;
  int block_num = matrix_size / block_size;
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  
  float sum[stride * stride]{0.0f};
  for (int b = 0; b < block_num; ++b) {
    // read elements to shared memory
    __shared__ float sa[block_size * block_size];
    __shared__ float sb[block_size * block_size];
    for (int i = 0; i < stride; ++i) {
      for (int j = 0; j < stride; ++j) {
        // (x, y) -> (x + i, y + i); (tx, ty) -> (tx + i, ty + i)
        int x0 = threadIdx.x * stride + i, y0 = threadIdx.y * stride + j;
        int x1 = tx * stride + i, y1 = ty * stride + j;
        sa[x0 + y0 * block_size] = A[x1 + (y1 % block_size + b * block_size) * m];
        sb[x0 + y0 * block_size] = B[(x1 % block_size + b * block_size) + y1 * k];
      }
    }
    __syncthreads();

    // compute
    for (int i = 0; i < stride; ++i) {
      for (int j = 0; j < stride; ++j) {
        int x0 = threadIdx.x * stride + i, y0 = threadIdx.y * stride + j;
        #pragma unroll
        for (int k = 0; k < block_size; ++k) {
          sum[i + j * stride] += sa[x0 + k * block_size] * sb[k + y0 * block_size];
        }
      }
    }
    __syncthreads();
  }

  for (int i = 0; i < stride; ++i) {
    for (int j = 0; j < stride; ++j) {
      int x1 = tx * stride + i, y1 = ty * stride + j;
      C[x1 + y1 * m] = sum[i + j * stride];
    }
  }
}

/**
一个线程计算 stride * stride 个元素；因此启动的 block 数量相应减少为 block / stride / stride
要求矩阵的元素数量大于 16 * stride(2) = 32
*/
template<int stride>
__global__ void BlockWithStrideAlignMatmulImpl(float *A, float *B, float *C, int m, int n, int k) {
  constexpr int block_size = 16 * stride;
  int matrix_size = m;
  int block_num = matrix_size / block_size;
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  
  float sum[stride * stride]{0.0f};
  for (int b = 0; b < block_num; ++b) {
    // read elements to shared memory
    __shared__ __align__(16) float sa[block_size * block_size];
    __shared__ __align__(16) float sb[block_size * block_size];
    for (int i = 0; i < stride; ++i) {
      for (int j = 0; j < stride; ++j) {
        // (x, y) -> (x + i, y + i); (tx, ty) -> (tx + i, ty + i)
        int x0 = threadIdx.x * stride + i, y0 = threadIdx.y * stride + j;
        int x1 = tx * stride + i, y1 = ty * stride + j;
        sa[x0 + y0 * block_size] = A[x1 + (y1 % block_size + b * block_size) * m];
        sb[x0 + y0 * block_size] = B[(x1 % block_size + b * block_size) + y1 * k];
      }
    }
    __syncthreads();

    // compute
    for (int i = 0; i < stride; ++i) {
      for (int j = 0; j < stride; ++j) {
        int x0 = threadIdx.x * stride + i, y0 = threadIdx.y * stride + j;
        #pragma unroll
        for (int k = 0; k < block_size; ++k) {
          sum[i + j * stride] += sa[x0 + k * block_size] * sb[k + y0 * block_size];
        }
      }
    }
    __syncthreads();
  }

  for (int i = 0; i < stride; ++i) {
    for (int j = 0; j < stride; ++j) {
      int x1 = tx * stride + i, y1 = ty * stride + j;
      C[x1 + y1 * m] = sum[i + j * stride];
    }
  }
}

} // namespace matmul

void NaiveMatmul(float *A, float *B, float *C, int m, int n, int k) {
  const auto &t = GetGridAndBlock(m, n);
  const auto &grid = std::get<0>(t);
  const auto &block = std::get<1>(t);
  {
    matmul::NaiveMatmulImpl<<<grid, block>>>(A, B, C, m, n, k);
    cudaDeviceSynchronize();
  }
}

void CublasMatmul(float *A, float *B, float *C, int m, int n, int k) {
  CublasHandle &cublas_handle = CublasHandle::GetInstance();
  auto handle = cublas_handle.GetCublasHandle();
  constexpr float alpha = 1.0f, beta = 0.0f;
  {
    /*
    lda, ldb, ldc:
    对于行优先，填写一行的元素个数；对于列优先，填写一列的元素个数。 ldx
    存在的原因是，这个矩阵可能是一个大矩阵的一个小矩阵，用来计算偏移量。
    以列优先为例，[i, j] 坐标对应的元素 offset = i + ldx * j (i/j 都从 0 开始)
    */
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k,
                &beta, C, m);
    cudaDeviceSynchronize();
  }
}

void BlockMatmul(float *A, float *B, float *C, int m, int n, int k) {
  const auto &t = GetGridAndBlock(m, n);
  const auto &grid = std::get<0>(t);
  const auto &block = std::get<1>(t);
  {
    matmul::BlockMatmulImpl<<<grid, block>>>(A, B, C, m, n, k);
    cudaDeviceSynchronize();
  }
}

void BlockWithStrideMatmul(float *A, float *B, float *C, int m, int n, int k) {
  constexpr int stride = 2;
  const auto &t = GetGridAndBlock(m, n);
  auto grid = std::get<0>(t);
  const auto &block = std::get<1>(t);
  grid.x /= stride;
  grid.y /= stride;
  {
    matmul::BlockWithStrideMatmulImpl<stride><<<grid, block>>>(A, B, C, m, n, k);
    cudaDeviceSynchronize();
  }
}

void BlockWithStrideAlignMatmul(float *A, float *B, float *C, int m, int n, int k) {
  constexpr int stride = 2;
  const auto &t = GetGridAndBlock(m, n);
  auto grid = std::get<0>(t);
  const auto &block = std::get<1>(t);
  grid.x /= stride;
  grid.y /= stride;
  {
    matmul::BlockWithStrideAlignMatmulImpl<stride><<<grid, block>>>(A, B, C, m, n, k);
    cudaDeviceSynchronize();
  }
}
