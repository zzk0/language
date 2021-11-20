# target 中的 public, private, interface

参考：https://zhuanlan.zhihu.com/p/82244559

这个其实写的更清楚: https://github.com/ttroy50/cmake-examples/tree/master/01-basic/C-static-library

`target_include_directories` 

- 如果设置了 PRIVATE, 那么使用这个 target 的其他 target-x, 就永不会用这些头文件; 只在当前 target 使用
- 如果设置了 INTERFACE, 那么这些头文件还会被加入到 target-x; 当前 target 不使用
- PUBLIC 结合以上两者

所以一般都是 private, 除非你要把 `target_include_directories` 里面的文件夹暴露给其他 target 使用

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

