project(cuda CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-arch=compute_35 -g -G -O3")
include_directories(./include)

add_subdirectory(0_hello_world)
