#include <cublas_v2.h>
#include <cuda_runtime.h>

/**
copy from: https://github.com/PaddlePaddle/CINN/blob/e824815fe145eaf2b86b9e1f20b19f05c8d0cd81/cinn/runtime/cuda/cuda_util.cc
*/
class CublasHandle {
 public:
  CublasHandle(const CublasHandle &) = delete;
  CublasHandle &operator=(const CublasHandle &) = delete;
  ~CublasHandle() {
    cublasDestroy(cuhandle);
  }
  static CublasHandle &GetInstance() {
    static CublasHandle instance;
    return instance;
  }
  cublasHandle_t &GetCublasHandle() { return cuhandle; }

 private:
  CublasHandle() {
    cublasCreate(&cuhandle);
  }
  cublasHandle_t cuhandle;
};
