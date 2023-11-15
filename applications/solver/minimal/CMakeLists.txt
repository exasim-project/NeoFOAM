# SPDX-License-Identifier: MPL-2.0
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_executable(minimalFoam 
	"minimal.cpp"
)

target_link_libraries(
	minimalFoam
	PRIVATE NeoFOAM::NeoFOAM
)
