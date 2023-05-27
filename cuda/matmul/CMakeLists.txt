CMAKE_MINIMUM_REQUIRED(VERSION 3.18)
project(matmul)
find_package(cuBLAS)

add_library(matmul_lib SHARED matmul.cu)
target_link_libraries(matmul_lib PRIVATE ${CUBLAS_LIBRARIES})

add_executable(${PROJECT_NAME} main.cu)
target_link_libraries(${PROJECT_NAME} PRIVATE cuda_lib matmul_lib)
