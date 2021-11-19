project(configure-file)

set(cf_example_MAJOR 1)
set(cf_example_VERSION 1.2)
message("cf_example PROJECT BINARY DIR: " ${PROJECT_BINARY_DIR})
configure_file(ver.h.in ${PROJECT_BINARY_DIR}/ver.h)

add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_BINARY_DIR})
