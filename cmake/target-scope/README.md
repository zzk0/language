# target 中的 public, private, interface

参考：https://zhuanlan.zhihu.com/p/82244559

## 指令

`target_include_directories`

`target_link_libraries`

`target_compile_options`

## 关系表
                    PRIVATE         INTERFACE       PUBLIC
hello_world.h       x               hello.h

hello_world.c       hello.h         x

PRIVATE: 引入了 hello_world.h，外面是看不到 hello.h 的。 

INTERFACE: 引入了 hello_world.h，可以使用 hello.h 的函数。hello_world.cpp 不使用 libhello.so 的功能，但可以用 hello.h 中定义的结构体等。

PUBLIC: PRIVATE + INTERFACE
