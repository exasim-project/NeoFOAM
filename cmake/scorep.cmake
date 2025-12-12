# Check whether scorep is present and set flag
function(scorep_check OUTVAR)
    # Paths based on expected bootstrap location
    set(SCOREP_PREFIX "${CMAKE_BINARY_DIR}/scorep")
    set(SCOREP_BIN    "${SCOREP_PREFIX}/bin")

    # Prepend potential Score-P install to PATH
    set(ENV{PATH} "${SCOREP_BIN}:$ENV{PATH}")

    find_program(SCOREP_INFO scorep-info HINTS ${SCOREP_BIN})

    if(SCOREP_INFO)
        message(STATUS "[Score-P] Found: ${SCOREP_INFO}")
        set(${OUTVAR} TRUE PARENT_SCOPE)
    else()
        message(STATUS "[Score-P] Not found → will bootstrap")
        set(${OUTVAR} FALSE PARENT_SCOPE)
    endif()
endfunction()


# Setup compiler wrappers AFTER Score-P is present
function(scorep_setup_compilers)
    set(SCOREP_PREFIX "${CMAKE_BINARY_DIR}/scorep")
    set(SCOREP_BIN    "${SCOREP_PREFIX}/bin")

    if(SCOREP_COMPILER_SUITE STREQUAL "gcc")
        message(STATUS "[Score-P] Using GNU Score-P wrappers")
        set(CMAKE_C_COMPILER   "${SCOREP_BIN}/scorep-gcc"  CACHE STRING "" FORCE)
        set(CMAKE_CXX_COMPILER "${SCOREP_BIN}/scorep-g++"  CACHE STRING "" FORCE)
    elseif(SCOREP_COMPILER_SUITE STREQUAL "clang")
        message(STATUS "[Score-P] Using Clang Score-P wrappers")
        set(CMAKE_C_COMPILER   "${SCOREP_BIN}/scorep-clang"   CACHE STRING "" FORCE)
        set(CMAKE_CXX_COMPILER "${SCOREP_BIN}/scorep-clang++" CACHE STRING "" FORCE)
    else()
        message(FATAL_ERROR "[Score-P] Unsupported compiler family: ${CMAKE_CXX_COMPILER_ID}")
    endif()

    set(ENV{SCOREP_WRAPPER_INSTRUMENTER_FLAGS} "--thread=pthread --mpp=none --kokkos")

    message(STATUS "[Score-P] Wrapper flags: $ENV{SCOREP_WRAPPER_INSTRUMENTER_FLAGS}")
endfunction()

