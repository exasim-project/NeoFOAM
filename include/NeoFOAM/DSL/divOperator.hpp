// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <iostream>

enum class EqTermType {
    temporalTerm,
    implicitTerm,
    explicitTerm
};

class PDEComponent
{
public:
    double coefficient;
    char variable;
    int exponent;

    PDEComponent(double coeff, char var, int exp);
    
    void print() const;
protected:
    EqTermType type;
};