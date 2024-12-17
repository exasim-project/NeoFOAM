# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

if(NEOFOAM_ENABLE_MPI_SUPPORT)
  add_subdirectory(mpi)
endif()

add_subdirectory(primitives)

neofoam_unit_test(dictionary)
neofoam_unit_test(tokenList)
neofoam_unit_test(inputs)
neofoam_unit_test(executor)
neofoam_unit_test(parallelAlgorithms)

# The registerClass test has to be created without linking against neofoam. Otherwise, it will pick
# up the factories registered in neofoam, which will make the test hard to maintain.
add_library(testRegister ${NeoFOAM_LIB_TYPE})
target_sources(testRegister PRIVATE "testRegister.cpp")
target_link_libraries(testRegister PRIVATE NeoFOAM_public_api cpptrace::cpptrace NeoFOAM)
if(NEOFOAM_ENABLE_MPI_SUPPORT)
  target_link_libraries(testRegister PRIVATE MPI::MPI_CXX)
endif()
if(WIN32)
  set_target_properties(
    testRegister
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/$<0:>
               LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/$<0:>
               ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/$<0:>)

else()
  set_property(TARGET testRegister PROPERTY RUNTIME_OUTPUT_DIRECTORY
                                            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)
endif()

add_executable(registerClass "registerClass.cpp")
target_link_libraries(registerClass PRIVATE Catch2::Catch2WithMain cpptrace::cpptrace
                                            NeoFOAM_public_api testRegister)
if(WIN32)
  set_target_properties(
    registerClass
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/$<0:>
               LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/$<0:>
               ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests/$<0:>)

else()
  set_property(TARGET registerClass PROPERTY RUNTIME_OUTPUT_DIRECTORY
                                             ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)
endif()

add_test(
  NAME registerClass
  COMMAND registerClass
  WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tests)
