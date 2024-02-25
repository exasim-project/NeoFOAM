// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#ifndef EQNTERM_H
#define EQNTERM_H

#include <iostream>
#include <memory>
#include <vector>

enum EqTermType {
    temporalTerm,
    implicitTerm,
    explicitTerm
};

class EqnTerm {
public:
    EqnTerm(EqTermType type) : type(type) {}
    virtual void explicitTerm() const = 0; // Pure virtual function
    virtual void implcitTerm() const = 0; // Pure virtual function
protected:
    EqTermType type;
    std::vector<std::unique_ptr<EqnTerm>> EqnTerms;
};

#endif // EQNTERM_Hchildren