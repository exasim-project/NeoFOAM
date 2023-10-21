add_library(NeoFOAM)

target_sources(NeoFOAM
  PUBLIC
    FILE_SET cxx_modules TYPE CXX_MODULES FILES
    "hello_world.cxx"
)
