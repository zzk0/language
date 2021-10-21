cmake_minimum_required(VERSION 3.18)
project(build-type)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message("Setting build type to 'RelWithDebInfo' as none was specified.")
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Chose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
            "MinSizeRel" "RelWithDebInfo")
endif ()

add_executable(hello_build_type main.cpp)
