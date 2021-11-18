add_executable(constructor main.cpp)
# 关闭编译器优化
# add_compile_definitions(-fno-elide-constructors)
set(CMAKE_CXX_FLAGS "-fno-elide-constructors ${CMAKE_CXX_FLAGS}")
