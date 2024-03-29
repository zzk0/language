#ifndef CUDA_INCLUDE_UTIL
#define CUDA_INCLUDE_UTIL

#include <iostream>
#include <chrono>
#include <cstdio>
#include <string>
#include <random>
#include <tuple>

#define CHECK(call)                                                            \
  do {                                                                         \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess) {                                                \
      printf("ERROR: %s:%d,", __FILE__, __LINE__);                             \
      printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

/**
T can be following
1. std::chrono::nanoseconds
2. std::chrono::microseconds
3. std::chrono::milliseconds
*/
template <typename T> class TimerRAII {
public:
  TimerRAII()
      : time_point_(std::chrono::system_clock::now()), format_str_("%d") {}

  TimerRAII(const std::string &format_str)
      : time_point_(std::chrono::system_clock::now()), format_str_(format_str) {
  }

  ~TimerRAII() {
    auto duration = std::chrono::system_clock::now() - time_point_;
    auto duration_in_t = std::chrono::duration_cast<T>(duration).count();
    printf(format_str_.c_str(), duration_in_t);
  }

private:
  std::chrono::high_resolution_clock::time_point time_point_;
  std::string format_str_;
};


/**
T can be following
1. std::chrono::nanoseconds
2. std::chrono::microseconds
3. std::chrono::milliseconds
*/
template <typename T> class Timer {
public:
  Timer() {}

  void Start() {
    start_time_point_ = std::chrono::system_clock::now();
  }

  void End() {
    end_time_point_ = std::chrono::system_clock::now();
  }

  int GetDuration() {
    return std::chrono::duration_cast<T>(end_time_point_ - start_time_point_).count();
  }

private:
  std::chrono::high_resolution_clock::time_point start_time_point_;
  std::chrono::high_resolution_clock::time_point end_time_point_;
};


template<typename T>
inline T RandomNumber(T low = -1, T high = 1) {
  static std::random_device rd;
  static std::mt19937 mt(rd());
  static std::uniform_real_distribution<float> dist(low, high);
  return static_cast<T>(dist(mt));
}

template<typename T>
inline void FillRandomNumber(std::vector<T> &data, size_t size, T low = -1, T high = 1) {
  for (size_t i = 0; i < size; ++i) {
    data[i] = RandomNumber<T>();
  }
}

inline std::tuple<dim3, dim3> GetGridAndBlock(int m) {
  dim3 block(128);
  int grid_x = m / 128 + (m % 128 != 0);
  dim3 grid(grid_x);
  return {grid, block};
}

inline std::tuple<dim3, dim3> GetGridAndBlock(int m, int n, const int block_x=16, const int block_y=16) {
  dim3 block(block_x, block_y);
  int grid_x = m / block_x + (m % block_x != 0);
  int grid_y = n / block_y + (n % block_y != 0);
  dim3 grid(grid_x, grid_y);
  return {grid, block};
}

/**
return true if two array has same elements, otherwise false
TODO: verbose not implement yet
*/
bool EqualCheckCUDA(float *dev_a, float *dev_b, int numel, bool on_exit=false, bool verbose=true);

inline std::vector<float> NewMatrix(int nx, int ny) {
  return std::vector<float>(nx * ny, 0);
}

inline void InitMatrix(std::vector<float> &matrix, int nx, int ny) {
  FillRandomNumber<float>(matrix, nx * ny);
}

inline void PrintMatrix(const std::vector<float> &matrix, int nx, int ny) {
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      std::cout << matrix[i * ny + j] << " ";
    }
    std::cout << std::endl;
  }
}

void PrintMatrixOnDevice(float *matrix, int nx, int ny);

#endif /* CUDA_INCLUDE_UTIL */
