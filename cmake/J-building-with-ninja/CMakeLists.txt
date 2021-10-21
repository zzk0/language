# building command:
# cmake .. -G Ninja
# ninja -v
# ninja is a build tool similar to make
# sudo apt-get install ninja-build

cmake_minimum_required(VERSION 3.18)
project(J-building-with-ninja)

find_package(Boost 1.46.1 REQUIRED COMPONENTS filesystem system)
if(Boost_FOUND)
    message(STATUS "${PROJECT_NAME}: boost found")
    include_directories(${Boost_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Cannot find Boost")
endif()

add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${BOOST_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
        PRIVATE
        ${Boost_SYSTEM_LIBRARY}
        ${Boost_FILESYSTEM_LIBRARY}
)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_auto_type)
message(STATUS "List of compile features: ${CMAKE_CXX_COMPILE_FEATURES}")
