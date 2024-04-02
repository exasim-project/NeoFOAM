add_test(NAME minimalFoamIsExecutable
         COMMAND ${CMAKE_BINARY_DIR}/applications/solver/minimal/minimalFoam)

add_test(NAME minimalFoamCanBeCalledWithHelp
         COMMAND ${CMAKE_BINARY_DIR}/applications/solver/minimal/minimalFoam
                 --help)

set_property(
  TEST minimalFoamCanBeCalledWithHelp
  PROPERTY PASS_REGULAR_EXPRESSION "A minimal example of a NeoFOAM application")

add_test(NAME minimalFoamCanBeCalledWithVersion
         COMMAND ${CMAKE_BINARY_DIR}/applications/solver/minimal/minimalFoam
                 --version)
set_property(
  TEST minimalFoamCanBeCalledWithVersion
  PROPERTY PASS_REGULAR_EXPRESSION
           "NeoFOAM version: [0-9].[0-9].[0-9] build: [a-b0-9]*")
