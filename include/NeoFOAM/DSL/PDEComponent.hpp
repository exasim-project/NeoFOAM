// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <iostream>
#include <vector>

enum class EqTermType {
    temporalTerm,
    implicitTerm,
    explicitTerm
};

class PDEComponent
{
public:

    PDEComponent(EqTermType eqnType);
    
    void print() const;

    virtual void explicitTerm(std::vector<double>& vec_out,const std::vector<double>& vec_in);

    virtual void implicitTerm(std::vector<double>& vec_out,const std::vector<double>& vec_in);
protected:
    EqTermType eqnType_;
};