# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

if(NEOFOAM_ENABLE_MPI_SUPPORT)
  neofoam_unit_test(fullDuplexCommBuffer MPI_SIZE 3)
  neofoam_unit_test(halfDuplexCommBuffer MPI_SIZE 3)
  neofoam_unit_test(operators MPI_SIZE 3)
endif()
