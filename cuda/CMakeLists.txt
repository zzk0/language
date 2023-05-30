project(cuda CXX C CUDA)
# set(CMAKE_CUDA_FLAGS "-arch=sm_70 -g -G -O3 -std=c++14 -Xptxas -v")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CUDA_FLAGS "-std=c++14 -maxrregcount=255")
else()
  set(CMAKE_CUDA_FLAGS "-O3 -std=c++14 -maxrregcount=255")
endif()

set(CMAKE_CUDA_ARCHITECTURES 70)  # default = 30
include_directories(/usr/local/cuda-10.2/include)
include_directories(./include)

add_subdirectory(lib)
add_subdirectory(0_hello_world)
add_subdirectory(reduce_sum)
add_subdirectory(matrix_sum)
add_subdirectory(matmul)
