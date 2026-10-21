# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

if(${NEOFOAM_WITH_MPI})
  neofoam_unit_test(distributedUnstructuredMesh setup_pressureVelocityCoupling MPI_SIZE 3 SOURCE
                    unstructuredMesh.cpp)
  neofoam_unit_test(distributedPressureVelocityCoupling setup_pressureVelocityCoupling MPI_SIZE 3
                    SOURCE pressureVelocityCoupling.cpp)
  neofoam_unit_test(distributedMomentum setup_pressureVelocityCoupling MPI_SIZE 3 SOURCE
                    momentum.cpp)
  neofoam_unit_test(distributedResidual setup_pressureVelocityCoupling MPI_SIZE 3 SOURCE
                    residual.cpp)
  neofoam_unit_test(distributedSnGrad setup_snGrad MPI_SIZE 3 SOURCE snGrad.cpp)
  neofoam_unit_test(distributedBound setup_pressureVelocityCoupling MPI_SIZE 3 SOURCE bound.cpp)
else()
  message(WARNING "Test skip building MPI based tests")
endif()
