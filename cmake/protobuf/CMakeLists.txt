project(protobuf-example)

find_package(Protobuf REQUIRED)
if (PROTOBUF_FOUND)
    message("protobuf found")
else()
    message(FATAL_ERROR "protobuf not found")
endif()

file(GLOB PROTO_FILES "${PROJECT_SOURCE_DIR}/*.proto")
message(${PROTO_FILES})
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
message("PROTO_SRCS: " ${PROTO_SRCS})
message("PROTO_HDRS: " ${PROTO_HDRS})

add_executable(${PROJECT_NAME} main.cpp ${PROTO_SRCS})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROTOBUF_LIBRARIES})
