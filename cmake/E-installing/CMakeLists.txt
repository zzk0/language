cmake_minimum_required(VERSION 3.18)
project(hello_install)

add_library(hello_installed_library SHARED src/Hello.cpp)
target_include_directories(hello_installed_library PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(hello_installed_binary src/main.cpp)
target_link_libraries(hello_installed_binary PRIVATE hello_installed_library)

# run: cmake .. -DCMAKE_INSTALL_PREFIX=/install/location

# This will install in ${CMAKE_INSTALL_PREFIX}/bin
install(TARGETS hello_installed_binary DESTINATION bin)
install(TARGETS hello_installed_library LIBRARY DESTINATION lib)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include DESTINATION include)
install(FILES cmake-examples.conf DESTINATION etc)
