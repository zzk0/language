CMAKE_MINIMUM_REQUIRED(VERSION 3.18)
project(reduce_sum)
add_executable(${PROJECT_NAME} main.cu)
# target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#                        --relocatable-device-code=true
#                        >)
# set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
