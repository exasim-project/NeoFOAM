# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2024 NeoFOAM authors

add_test(NAME field_test COMMAND test_Field)
add_executable(test_Field "test_Field.cpp")
target_link_libraries(test_Field PRIVATE neofoam_catch_main)

add_test(NAME domain_field_test COMMAND test_DomainField)
add_executable(test_DomainField "test_DomainField.cpp")
target_link_libraries(test_DomainField PRIVATE neofoam_catch_main)
