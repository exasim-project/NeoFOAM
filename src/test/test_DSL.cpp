// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "gtest/gtest.h"
#include <vector>
#include <iostream>
#include "NeoFOAM/DSL/PDEComponent.hpp"
#include "NeoFOAM/DSL/laplacianOperator.hpp"
#include "NeoFOAM/DSL/PDEExpression.hpp"

#include <iostream>

class laplacianTest: public laplacianMethod {
public:
    virtual void explicitTerm(std::vector<double>& vec_out,const std::vector<double>& vec_in) override {
        for (int i = 0; i < vec_in.size(); i++)
        {
            vec_out[i] = vec_in[i];
        }
    }
    
    virtual void implicitTerm(std::vector<double>& vec_out,const std::vector<double>& vec_in) override {
        for (int i = 0; i < vec_in.size(); i++)
        {
            vec_out[i] = vec_in[i];
        }
    }
};


TEST(DSL, check_number_terms) {
    PDEComponent t1(EqTermType::temporalTerm);
    PDEComponent e1(EqTermType::explicitTerm);
    PDEComponent i1(EqTermType::implicitTerm);
    laplacianOperator lap1(std::make_unique<laplacianTest>(),EqTermType::implicitTerm);

    PDEExpression equation = t1 + e1 + i1 + lap1;
    EXPECT_EQ(equation.terms.size(), 4);

}

TEST(DSL, check_explicit_term) {
    laplacianOperator lap1(std::make_unique<laplacianTest>(),EqTermType::implicitTerm);

    PDEExpression equation = lap1;
    std::vector<double> in = {1,2,3,4};
    std::vector<double> out = {0,0,0,0};
    equation.terms[0].explicitTerm(out,in);
    EXPECT_EQ(out[0], 1);
    EXPECT_EQ(equation.terms.size(), 1);

}