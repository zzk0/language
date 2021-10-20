cmake_minimum_required(VERSION 3.18)
project(static-library)
message(STATUS "C-static-library: ${PROJECT_SOURCE_DIR}")
# 如果不设置 project, 那么 project source dir 会是最顶层的那个

add_library(hello_library STATIC src/Hello.cpp)
target_include_directories(hello_library PUBLIC ${PROJECT_SOURCE_DIR}/include)
# PUBLIC 的作用是设置作用域, PUBLIC 表示可以用于编译 library 的时候, 以及任何使用了这个 library 的地方(含有以下两者)
# INTERFACE 作用域只在于任何用到了这个 Library 的地方
# PRIVATE 作用域只在 target 的 include

add_executable(hello_binary src/main.cpp)
target_link_libraries(hello_binary PRIVATE hello_library)
