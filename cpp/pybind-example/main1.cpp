#include "pybind11/pybind11.h"

int multiply(int i, int j) {
  return  i * j;
}

// 编译链接的时候会报错, multiple definition of `PyInit_cmake_example'
//PYBIND11_MODULE(cmake_example, m) {
//  m.def("multiply", &multiply);
//}
