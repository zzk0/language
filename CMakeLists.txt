cmake_minimum_required(VERSION 3.18)
project(cmake_cpp_cuda)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-module)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)
option(BUILD_CPP_PROTOBUF OFF)
option(BUILD_PYBIND OFF)

if (BUILD_CPP_PROTOBUF)
    message("-------- build cpp protobuf")
endif()

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/third_party)
include(third_party)

add_subdirectory(cmake)
add_subdirectory(cpp)
add_subdirectory(cuda)
