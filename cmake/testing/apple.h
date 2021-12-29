#ifndef CMAKE_CPP_CUDA_CMAKE_TESTING_APPLE_H_
#define CMAKE_CPP_CUDA_CMAKE_TESTING_APPLE_H_

#include "iostream"

class Apple {
 private:
  int size;
 public:
  Apple() {
    std::cout << "default constructor" << std::endl;
  }
  void Print() {
    std::cout << size << std::endl;
  }
};

Apple GetApple();

#endif //CMAKE_CPP_CUDA_CMAKE_TESTING_APPLE_H_
