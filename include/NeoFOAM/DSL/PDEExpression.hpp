// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <vector>
#include <unordered_map>
#include "PDEComponent.hpp"
#include <memory>

class PDEExpression
{
public:
    // std::vector<std::unique_ptr<PDEComponent>> terms;
    std::vector<PDEComponent> terms;
    // std::unordered_map<EqTermType, PDEComponent> terms2;

    PDEExpression(const PDEComponent &term);
    void addTerm(const PDEComponent &term);
    void print() const;
    PDEExpression operator+(const PDEExpression &rhs);
    PDEExpression operator+(const PDEComponent &rhs);
};

PDEExpression operator+(const PDEComponent &lhs, const PDEComponent &rhs);