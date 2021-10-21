cmake_minimum_required(VERSION 3.18)
project(H-third-party-library)

find_package(Boost 1.46.1 REQUIRED COMPONENTS filesystem system)
if(Boost_FOUND)
    message(STATUS "${PROJECT_NAME}: boost found")
    include_directories(${Boost_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "Cannot find Boost")
endif()

add_executable(third_party_include main.cpp)
target_include_directories(third_party_include PRIVATE ${BOOST_INCLUDE_DIRS})
target_link_libraries(third_party_include
        PRIVATE
        ${Boost_SYSTEM_LIBRARY}
        ${Boost_FILESYSTEM_LIBRARY}
)

