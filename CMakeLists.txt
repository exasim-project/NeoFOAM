cmake_minimum_required(VERSION 3.26.0)

project(
  NeoFOAM
  LANGUAGES C CXX
  DESCRIPTION
  "A prototype implementation of OpenFOAM in C++ 20")

set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API "2182bf5c-ef0d-489a-91da-49dbc3090d2a")
set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)
string(CONCAT CMAKE_EXPERIMENTAL_CXX_SCANDEP_SOURCE
  "${CMAKE_CXX_COMPILER_CLANG_SCAN_DEPS}"
  " -format=p1689"
  " --"
  " <CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS>"
  " -x c++ <SOURCE> -c -o <OBJECT>"
  " -MT <DYNDEP_FILE>"
  " -MD -MF <DEP_FILE>"
  " > <DYNDEP_FILE>")
set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FORMAT "clang")
set(CMAKE_EXPERIMENTAL_CXX_MODULE_MAP_FLAG "@<MODULE_MAP_FILE>")


# Default to C++ extensions being off. Clang's modules support have trouble
# with extensions right now.
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 20)
add_library(NeoFOAM)
target_sources(NeoFOAM
  PUBLIC
    FILE_SET cxx_modules TYPE CXX_MODULES FILES
    src/hello_world.cxx
)
