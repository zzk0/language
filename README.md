# Programming Language

A project for practicing programming language CMake, Cpp, Cuda.

# cmake

reference: https://github.com/ttroy50/cmake-examples

[这篇文章](https://zhuanlan.zhihu.com/p/267803605) 包含了大部分常见用法。

# cpp

[modern cpp feature](https://github.com/AnthonyCalandra/modern-cpp-features): Modern CPP 新特性

[可变模板参数函数的参数包展开](cpp/variadic-template-1): 介绍两种展开可变模板参数函数的参数包的方法, 一种是递归调用, 一种是利用逗号表达式

[实现一个线程池](cpp/thread-pool): 实现参考[这篇](https://zhuanlan.zhihu.com/p/61464921)

[实现 JIT !!!](cpp/jit): 看[这篇文章](https://blog.reverberate.org/2012/12/hello-jit-world-joy-of-simple-jits.html)

reference: https://changkun.de/modern-cpp/zh-cn/00-preface/

# cuda

reference: https://github.com/Tony-Tan/CUDA_Freshman

https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

一份官方的中文资料: https://www.nvidia.cn/docs/IO/51635/NVIDIA_CUDA_Programming_Guide_1.1_chs.pdf

《CUDA C 编程权威指南》

## Build

To build the project, you need to specify the location of nvcc:

```
-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
```

