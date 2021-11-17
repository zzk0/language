find_package(Threads)
add_executable(smart-pointer main.cpp)
target_link_libraries(smart-pointer ${CMAKE_THREAD_LIBS_INIT})
