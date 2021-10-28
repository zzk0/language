CMAKE_MINIMUM_REQUIRED(VERSION 3.18)
project(pybind-example LANGUAGES CXX)

# https://cmake.org/cmake/help/v3.11/module/FetchContent.html
# FetchContent 是在配置的时候下载, 而 ExternalProject_Add 是在构建的时候
# 这样在构建的时候就可以使用需要的内容, 所谓的 FetchContent, 比如下面的 SOURCE_DIR 变量
include(FetchContent)

# 如果 CMake 项目分层, 那么只用第一个
FetchContent_Declare(
        pybind11_sources
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.6.0
)

# 当调用 FetchContent_Populate 的时候, 会有一些全局变量保存了属性 properties
# 使用下面的指令来获取这些变量, 比如 XXX_POPULATED, XXX_SOURCE_DIR, XXX_BINARY_DIR
FetchContent_GetProperties(pybind11_sources)

# 使用 XXX_POPULTED 可以避免重复下载, POPULTED 具有填充的意思
if (NOT pybind1_sources_POPULATED)
    FetchContent_Populate(pybind11_sources)
    message(${pybind11_sources_SOURCE_DIR})
    message(${pybind11_sources_BINARY_DIR})
    add_subdirectory(
            ${pybind11_sources_SOURCE_DIR}
            ${pybind11_sources_BINARY_DIR}
    )
endif ()

# 写法参照这里: https://github.com/pybind/cmake_example
pybind11_add_module(cmake_example main.cpp main1.cpp)
target_compile_definitions(cmake_example PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

# 使用的时候注意 python 版本, 如果 conda 的 python 版本没有对应上, 不会使用生成的动态链接库
