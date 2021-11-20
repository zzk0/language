project(protobuf-cpp)

set(THIRD_PARTY_DIR "${PROJECT_BINARY_DIR}/third_party_install"
        CACHE PATH "Where to install third party headers and libs")

# include 指令里面的 set 操作的变量作用域就是在这个文件,
# 可以类比 c++ 的 include 相当于把那里面的东西 include 进来
set(cmake_dir ${PROJECT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${cmake_dir})
list(APPEND CMAKE_MODULE_PATH ${cmake_dir}/third_party)

# 最好设置代理, 需要从 github 下载源代码
include(zlib)
include(protobuf)
include(proto2cpp)

file(GLOB PROTO_FILES ${PROJECT_SOURCE_DIR}/*.proto)
foreach(proto_name ${PROTO_FILES})
    file(RELATIVE_PATH proto_rel_name ${PROJECT_SOURCE_DIR} ${proto_name})
    list(APPEND REL_PROTO_FILES ${proto_rel_name})
endforeach()
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${PROJECT_SOURCE_DIR} ${REL_PROTO_FILES})
add_library(proto_lib STATIC ${PROTO_SRCS} ${PROTO_HDRS})
# 这里设置为 PUBLIC 是因为在链接生成 exe 的时候, 需要这些 include
# include 的本质就是将那些东西复制进来, 所以 main.cpp 上面就会 include PROTOBUF_INCLUDE_DIR
# 因此需要设置为 PUBLIC 才行
target_include_directories(proto_lib PUBLIC ${PROTOBUF_INCLUDE_DIR})
target_link_libraries(proto_lib PRIVATE ${PROTOBUF_STATIC_LIBRARIES})

add_executable(${PROJECT_NAME} main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE proto_lib)
