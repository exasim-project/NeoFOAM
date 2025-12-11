# ================================================================
#   ScorePBootstrap.cmake
#   Bootstraps Score-P if not yet present, then stops configuration
# ================================================================

include(ExternalProject)

# Detect compiler family for Score-P
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(SCOREP_COMPILER_SUITE gcc)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(SCOREP_COMPILER_SUITE clang)
else()
    message(FATAL_ERROR "[Score-P] Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()
message(STATUS "[Score-P] Compiler: ${SCOREP_COMPILER_SUITE}")

ExternalProject_Add(scorep_build
    SOURCE_DIR     ${CMAKE_BINARY_DIR}/_deps/scorep-src
    URL            https://perftools.pages.jsc.fz-juelich.de/cicd/scorep/tags/scorep-9.3/scorep-9.3.tar.gz
    URL_HASH       SHA256=5498b31b1d6c04b08a9d408320a7515e884538d248de58b6dd11b48c8f364112

    CONFIGURE_COMMAND
        env CC=${CMAKE_C_COMPILER}
            CXX=${CMAKE_CXX_COMPILER} 
        ${CMAKE_BINARY_DIR}/_deps/scorep-src/configure
            --prefix=${SCOREP_PREFIX}
            --with-libgotcha=download
            --with-libbfd=download
            --with-cuda
	    --with-nocross-compiler-suite=${SCOREP_COMPILER_SUITE}

    BUILD_COMMAND   make -j
    INSTALL_COMMAND make install
)

