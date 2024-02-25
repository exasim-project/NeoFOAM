#include <iostream>
#include <vector>
#include <map>
#include <memory>

// ```mermaid
// ---
// title: detailed design of the operator
// ---
// erDiagram
//     temporalTerm ||--|| EqnTerm: derived
//     implicitTerm ||--|| EqnTerm: derived
//     explicitTerm ||--|| EqnTerm: derived
//     explicitTerm {
//       EqTermType explicit
//       field scale
//       function scale
//     }
//     implicitTerm {
//       EqTermType implicit
//       field scale
//       function scale
//     }
//     temporalTerm {
//       EqTermType temporal
//       field scale
//       function scale
//     }
//     EqnTerm {
//       Enum EqTermType
//     }
//     EqnTerm ||--|| geometricalField: "requires"
//     EqnTerm ||--|| EnumName:  " "
//     EnumName {
//         string temporalTerm
//         string implicitTerm
//         string explicitTerm
//     }
//     EqnTerms {
//       vector[EqnTerm] eqnterms_
//       ADDoperator appendToList
//       SUBoperator appendToList
//     }
//     EqnTerms ||--|| fvmddT : "returns EqnTerms: temporalTerm"
//     EqnTerms ||--|| fvmdiv : "returns EqnTerms: implicitTerm"
//     EqnTerms ||--|| fvmLaplacian : "returns EqnTerms: implicitTerm"
//     EqnTerms ||--|| fvcdiv : "returns EqnTerms:  explicitTerm"
// ```
// can you write the classes from the class diagram above in cpp?
// you can use the following code as a starting point

// Base class for expressions
class EqnTerm { // EqnTerm
public:
    enum EqTermType { // EnumName
        temporalTerm,
        implicitTerm,
        explicitTerm
    };

    EqnTerm(EqTermType t) : type(t) {}
    virtual ~EqnTerm() = default;
    virtual void print() const = 0;

    EqTermType type;
};

// Derived class for Category 1 expressions

// Enumeration for expression categories
enum class EqTermType {
    temporalTerm,
    implicitTerm,
    explicitTerm
};

// Base class for expressions
class Expression {
public:
    EqTermType type;

    explicit Expression(EqTermType t) : type(t) {}
    virtual ~Expression() = default;
    virtual void print() const = 0;
};

// Derived class for Category 1 expressions
class temporalTerm : public Expression {
public:
    temporalTerm() : Expression(EqTermType::temporalTerm) {}

    void print() const override {
        std::cout << "Category 1 expression" << std::endl;
    }
};

// Derived class for Category 2 expressions
class implicitTerm : public Expression {
public:
    implicitTerm() : Expression(EqTermType::implicitTerm) {}

    void print() const override {
        std::cout << "Category 2 expression" << std::endl;
    }
};

// Derived class for Category 3 expressions
class explicitTerm : public Expression {
public:
    explicitTerm() : Expression(EqTermType::explicitTerm) {}

    void print() const override {
        std::cout << "Category 3 expression" << std::endl;
    }
};

// Expression Manager Class
class ExpressionManager {
private:
    std::map<EqTermType, std::vector<std::shared_ptr<Expression>>> expressionsMap;

public:
    // Add expressions to the manager
    void addExpression(std::shared_ptr<Expression> expr) {
        expressionsMap[expr->type].push_back(expr);
    }

    // Get the expressions of a specific category
    const std::vector<std::shared_ptr<Expression>>& getExpressions(EqTermType type) const {
        return expressionsMap.at(type);
    }

    // Print all expressions
    void printExpressions() const {
        for (const auto& pair : expressionsMap) {
            for (const auto& expr : pair.second) {
                expr->print();
            }
        }
    }
};

int main() {
    ExpressionManager manager;

    // Adding expressions to the manager
    manager.addExpression(std::make_shared<temporalTerm>());
    manager.addExpression(std::make_shared<implicitTerm>());
    manager.addExpression(std::make_shared<explicitTerm>());

    // Printing expressions by category
    manager.printExpressions();

    // If you need to access the vectors directly:
    const auto& cat1Expressions = manager.getExpressions(EqTermType::temporalTerm);
    const auto& cat2Expressions = manager.getExpressions(EqTermType::implicitTerm);
    const auto& cat3Expressions = manager.getExpressions(EqTermType::explicitTerm);

    // Print Category 1 expressions
    for (const auto& expr : cat1Expressions) {
        expr->print();
    }

    // No need to delete expressions; shared_ptr takes care of memory management.
    return 0;
}

