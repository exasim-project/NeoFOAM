# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

add_executable(minimalFoam 
	"minimal.cpp"
)

target_link_libraries(
	minimalFoam
	PRIVATE NeoFOAM
)
