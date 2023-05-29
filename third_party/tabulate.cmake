# tabulate is a header-only library. Just include the directory.
include(ExternalProject)

set(TABULATE_ROOT_DIR ${THIRD_PARTY_DIR}/tabulate)
set(TABULATE_INCLUDE_DIR ${THIRD_PARTY_DIR}/tabulate/src/tabulate/include)

set(TABULATE_URL https://github.com/p-ranav/tabulate/archive/refs/tags/v1.4.zip)

include_directories(${TABULATE_INCLUDE_DIR})

ExternalProject_Add(
    tabulate
    URL ${TABULATE_URL}
    PREFIX ${TABULATE_ROOT_DIR}
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND "")
