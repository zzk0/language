CMAKE_MINIMUM_REQUIRED(VERSION 3.18)
project(cuda_lib)
add_library(${PROJECT_NAME} SHARED util.cu)
