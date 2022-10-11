#ifndef CUDA_INCLUDE_UTIL
#define CUDA_INCLUDE_UTIL

#include <chrono>
#include <cstdio>
#include <string>
#include <random>

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

#endif /* CUDA_INCLUDE_UTIL */
