cmake_minimum_required(VERSION 3.18)
project(hello_headers)
message(STATUS "B-hello-headers: " ${PROJECT_SOURCE_DIR})

set(SOURCES src/Apple.cpp src/main.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
