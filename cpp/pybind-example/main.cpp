#include "pybind11/pybind11.h"
#include "iostream"

int add(int i, int j) {
  return i + j;
}

int substrate(int i, int j) {
  return i - j;
}

class Apple {
 public:
  void setSize(int x) {
    size_ = x;
  }

  int getSize() {
    return size_;
  }
 private:
  int size_;
};

namespace py = pybind11;

PYBIND11_MODULE(cmake_example, m) {
  std::cout << "pybind11 run" << std::endl;  // 这个会在载入动态链接库的时候调用!

  m.def("add", &add);

  py::class_<Apple>(m, "Apple")
      .def(py::init())
      .def("setSize", &Apple::setSize)
      .def("getSize", &Apple::getSize);

  py::module_ new_module = m.def_submodule("util");
  new_module.def("substrate", &substrate);
}
