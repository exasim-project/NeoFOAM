# Check whether scorep is present and set flag
function(scorep_check OUTVAR)
    # Paths based on expected bootstrap location
    set(SCOREP_PREFIX "${PROJECT_SOURCE_DIR}/build/scorep" CACHE STRING "Installation prefix for Score-P")
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
