#ifndef CUDA_INCLUDE_SMART_POINTER
#define CUDA_INCLUDE_SMART_POINTER

#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>

#include <cuda_runtime.h>

#include "util.h"

template <typename T> class DeviceUniquePointer {
public:
  DeviceUniquePointer(size_t bytes) : address_(nullptr), bytes_(bytes) {
    CHECK(cudaMalloc((T **)&address_, bytes));
  }

  DeviceUniquePointer(T *host_address, size_t elements) {
    bytes_ = sizeof(T) * elements;
    CHECK(cudaMalloc((T **)&address_, bytes_));
    CHECK(cudaMemcpy(address_, host_address, bytes_, cudaMemcpyHostToDevice));
  }

  ~DeviceUniquePointer() { CHECK(cudaFree(address_)); }

  DeviceUniquePointer(const DeviceUniquePointer &pointer) = delete;
  DeviceUniquePointer &operator=(const DeviceUniquePointer &pointer) = delete;

public:
  T *Get() { return address_; }
  std::unique_ptr<T, void (*)(T *)> GetHostPtr() {
    T *host_address = (T *)malloc(bytes_);
    CHECK(cudaMemcpy(host_address, address_, bytes_, cudaMemcpyDeviceToHost));
    return std::unique_ptr<T, void (*)(T *)>(host_address,
                                             [](T *ptr) { free(ptr); });
  }

private:
  T *address_; // this is a device address
  size_t bytes_;
};

inline float RandomFloat() {
  static std::random_device rd;
  static std::mt19937 mt(rd());
  static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  return dist(mt);
}

#endif /* CUDA_INCLUDE_SMART_POINTER */
