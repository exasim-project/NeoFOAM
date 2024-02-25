class fvmddT : public EqnTerm {
public:
    fvmddT() : EqnTerm(temporalTerm) {}
    void print() const override {
        std::cout << "fvmddT: temporalTerm" << std::endl;
    }
};

class fvmdiv : public EqnTerm {
public:
    fvmdiv() : EqnTerm(implicitTerm) {}
    void print() const override {
        std::cout << "fvmdiv: implicitTerm" << std::endl;
    }
};

class fvmLaplacian : public EqnTerm {
public:
    fvmLaplacian() : EqnTerm(implicitTerm) {}
    void print() const override {
        std::cout << "fvmLaplacian: implicitTerm" << std::endl;
    }
};

class fvcdiv : public EqnTerm {
public:
    fvcdiv() : EqnTerm(explicitTerm) {}
    void print() const override {
        std::cout << "fvcdiv: explicitTerm" << std::endl;
    }
};