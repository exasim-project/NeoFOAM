#include "NeoFOAM/DSL/PDEComponent.hpp"

PDEComponent::PDEComponent(EqTermType eqnType) : eqnType_(eqnType) {}


void PDEComponent::explicitTerm(std::vector<double>& vec_out,const std::vector<double>& vec_in)
{

}

void PDEComponent::implicitTerm(std::vector<double>& vec_out,const std::vector<double>& vec_in)
{

}
