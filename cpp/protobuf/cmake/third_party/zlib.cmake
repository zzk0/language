include (ExternalProject)

set(ZLIB_INSTALL ${THIRD_PARTY_DIR}/zlib)
set(ZLIB_INCLUDE_DIR ${ZLIB_INSTALL}/include)
set(ZLIB_LIBRARY_DIR ${ZLIB_INSTALL}/lib)
set(ZLIB_URL https://github.com/madler/zlib/archive/v1.2.8.tar.gz)

if(WIN32)
    set(ZLIB_LIBRARY_NAMES zlibstaticd.lib)
else()
    if(BUILD_SHARED_LIBS)
        if("${CMAKE_SHARED_LIBRARY_SUFFIX}" STREQUAL ".dylib")
            set(ZLIB_LIBRARY_NAMES libz.dylib)
        elseif("${CMAKE_SHARED_LIBRARY_SUFFIX}" STREQUAL ".so")
            set(ZLIB_LIBRARY_NAMES libz.so)
        else()
            message(FATAL_ERROR "${CMAKE_SHARED_LIBRARY_SUFFIX} not support for zlib")
        endif()
    else()
        set(ZLIB_LIBRARY_NAMES libz.a)
    endif()
endif()

foreach(LIBRARY_NAME ${ZLIB_LIBRARY_NAMES})
    list(APPEND ZLIB_STATIC_LIBRARIES ${ZLIB_LIBRARY_DIR}/${LIBRARY_NAME})
endforeach()

set(ZLIB_HEADERS
        "${ZLIB_INSTALL}/include/zconf.h"
        "${ZLIB_INSTALL}/include/zlib.h"
        )

ExternalProject_Add(zlib
        PREFIX zlib
        URL ${ZLIB_URL}
        URL_MD5 1eabf2698dc49f925ce0ffb81397098f
        UPDATE_COMMAND ""
        BUILD_IN_SOURCE 1
        BUILD_BYPRODUCTS ${ZLIB_STATIC_LIBRARIES}
        CMAKE_CACHE_ARGS
        -DCMAKE_C_COMPILER_LAUNCHER:STRING=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER:STRING=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
        -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
        -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
        -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_INSTALL_PREFIX:STRING=${ZLIB_INSTALL}
        )

add_library(zlib_imported UNKNOWN IMPORTED)
set_property(TARGET zlib_imported PROPERTY IMPORTED_LOCATION "${ZLIB_STATIC_LIBRARIES}")
