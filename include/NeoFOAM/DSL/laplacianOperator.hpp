// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <iostream>
#include "PDEComponent.hpp"
#include <memory>
#include <vector>

class laplacianMethod
{
public:
    // Virtual destructor to allow for proper cleanup of derived classes
    virtual ~laplacianMethod() {}

    virtual void explicitTerm(std::vector<double>& vec_out,const std::vector<double>& vec_in) = 0;

    virtual void implicitTerm(std::vector<double>& vec_out,const std::vector<double>& vec_in) = 0;
};

class laplacianOperator
    : public PDEComponent
{
public:
    laplacianOperator(std::unique_ptr<laplacianMethod> lapMethod, EqTermType eqnType);

    void explicitTerm(std::vector<double>& vec_out,const std::vector<double>& vec_in);

    void implicitTerm(std::vector<double>& vec_out,const std::vector<double>& vec_in);

protected:
    EqTermType eqnType_;
    std::unique_ptr<laplacianMethod> lapMethod_;
};