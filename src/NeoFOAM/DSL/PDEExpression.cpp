#include "NeoFOAM/DSL/PDEExpression.hpp"
#include <iostream>


PDEExpression::PDEExpression(const PDEComponent &term)
{
    // terms.push_back(std::make_unique<PDEComponent>(term));
}


void PDEExpression::addTerm(const PDEComponent &term)
{
    // terms.push_back(std::make_unique<PDEComponent>(term));
}


// adds two equations together
PDEExpression PDEExpression::operator+(const PDEExpression &rhs)
{
    PDEExpression result(*this);
    for (const auto &term : rhs.terms)
    {
        // result.terms.push_back(std::make_unique<PDEComponent>(std::move(&term)));
    }
    return result;
}

// adds a term to an equation
PDEExpression PDEExpression::operator+(const PDEComponent &rhs)
{
    PDEExpression result(*this);
    result.addTerm(rhs);    
    return result;
}

// Operator+ to combine two Terms into an PDEExpression
PDEExpression operator+(const PDEComponent &lhs, const PDEComponent &rhs)
{
    PDEExpression result(lhs);
    result.addTerm(rhs);  
    return result;
}