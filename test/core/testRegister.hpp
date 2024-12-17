// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors
#pragma once

#include "NeoFOAM/core/runtimeSelectionFactory.hpp"

class BaseClass : public NeoFOAM::RuntimeSelectionFactory<BaseClass, NeoFOAM::Parameters<>>
{

public:

    BaseClass();

    static std::string name();
};

class BaseClass2 : public NeoFOAM::RuntimeSelectionFactory<BaseClass2, NeoFOAM::Parameters<>>
{

public:

    BaseClass2();

    static std::string name();
};

class DerivedClass : public BaseClass::Register<DerivedClass>
{
public:

    DerivedClass();

    static std::string name();

    static std::string doc();

    static std::string schema();
};

class DerivedClass2 : public BaseClass2::Register<DerivedClass2>
{
public:

    DerivedClass2();

    static std::string name();

    static std::string doc();

    static std::string schema();
};
