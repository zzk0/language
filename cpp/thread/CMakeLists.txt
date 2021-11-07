CMAKE_MINIMUM_REQUIRED(VERSION 3.18)
project(thread)

find_package(Threads)
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${CMAKE_THREAD_LIBS_INIT})
