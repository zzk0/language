add_library(apple_lib apple.h apple.cpp)
message(STATUS "-------------------------")
message(STATUS ${apple_lib})
message(STATUS "-------------------------")
set(apple_lib -Wl,--whole-archive apple_lib -Wl,--no-whole-archive)
message(STATUS ${apple_lib})

add_executable(apple_exe main.cpp)
target_link_libraries(apple_exe ${apple_lib})
