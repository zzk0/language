cmake_minimum_required(VERSION 3.18)
project(cmake_cpp_cuda)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_CPP_PROTOBUF OFF)
option(BUILD_PYBIND OFF)

if (BUILD_CPP_PROTOBUF)
    message("-------- build cpp protobuf")
endif()

add_subdirectory(cmake)
add_subdirectory(cpp)
add_subdirectory(cuda)
