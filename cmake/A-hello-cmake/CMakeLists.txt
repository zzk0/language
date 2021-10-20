cmake_minimum_required(VERSION 3.18)
project(hello_cmake)
add_executable(hello_cmake main.cpp)
message(STATUS "A-hello-cmake: ${CMAKE_BINARY_DIR}")
