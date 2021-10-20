cmake_minimum_required(VERSION 3.18)
project(shared_library)

add_library(hello_shared_library SHARED src/Hello.cpp)
add_library(hello::library ALIAS hello_shared_library)
# target_include_directories(hello::library PUBLIC ${PROJECT_SOURCE_DIR}/include)
# alias cannot be used
target_include_directories(hello_shared_library PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(hello_shared_binary src/main.cpp)
target_link_libraries(hello_shared_binary PRIVATE hello::library)
