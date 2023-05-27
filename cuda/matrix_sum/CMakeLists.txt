CMAKE_MINIMUM_REQUIRED(VERSION 3.18)
project(matrix_sum)
add_executable(${PROJECT_NAME} main.cu)
target_link_libraries(${PROJECT_NAME} cuda_lib)
# nvprof ./cuda/matrix_sum/matrix_sum 
