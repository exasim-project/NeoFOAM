#include "EqnTerm.h"

fvmddT::fvmddT() : EqnTerm(temporalTerm) {}

void fvmddT::print() const {
    std::cout << "fvmddT: temporalTerm" << std::endl;
}

fvmdiv::fvmdiv() : EqnTerm(implicitTerm) {}

void fvmdiv::print() const {
    std::cout << "fvmdiv: implicitTerm" << std::endl;
}

fvmLaplacian::fvmLaplacian() : EqnTerm(implicitTerm) {}

void fvmLaplacian::print() const {
    std::cout << "fvmLaplacian: implicitTerm" << std::endl;
}

fvcdiv::fvcdiv() : EqnTerm(explicitTerm) {}

void fvcdiv::print() const {
    std::cout << "fvcdiv: explicitTerm" << std::endl;
}