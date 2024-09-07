// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#include "testRegister.hpp"

BaseClass::BaseClass() {}

std::string BaseClass::name() { return "BaseClass"; }


BaseClass2::BaseClass2() {}

std::string BaseClass2::name() { return "BaseClass2"; }

DerivedClass::DerivedClass() {}

std::string DerivedClass::name() { return "DerivedClass"; }

std::string DerivedClass::doc() { return "DerivedClass documentation"; }

std::string DerivedClass::schema() { return "DerivedClass schema"; }


DerivedClass2::DerivedClass2() {}

std::string DerivedClass2::name() { return "DerivedClass2"; }

std::string DerivedClass2::doc() { return "DerivedClass2 documentation"; }

std::string DerivedClass2::schema() { return "DerivedClass2 schema"; }
