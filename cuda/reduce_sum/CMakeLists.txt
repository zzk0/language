CMAKE_MINIMUM_REQUIRED(VERSION 3.18)
project(reduce_sum)
set(CMAKE_CUDA_FLAGS "-arch=compute_35 -g -G -std=c++14")
add_executable(${PROJECT_NAME} main.cu)
