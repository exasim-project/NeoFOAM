// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/DSL/laplacianOperator.hpp"

laplacianOperator::laplacianOperator(std::unique_ptr<laplacianMethod> lapMethod, EqTermType eqnType)
    : PDEComponent(eqnType), lapMethod_(std::move(lapMethod))
{
}

void laplacianOperator::explicitTerm(std::vector<double>& vec_out,const std::vector<double>& vec_in)
{
    lapMethod_->explicitTerm(vec_out, vec_in);
}

void laplacianOperator::implicitTerm(std::vector<double>& vec_out,const std::vector<double>& vec_in)
{
    lapMethod_->implicitTerm(vec_out, vec_in);
}
