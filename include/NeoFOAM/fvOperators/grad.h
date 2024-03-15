

#pragma once
#include <array>
#include <vector>
#include "fields/Field.hpp"

using scalar = double;
using vector = std::array<scalar, 3>;
using Field = std::vector<scalar>;
using face = int; // this is wrong
// What are the class responsibilities?

namespace NeoFOAM
{
    struct Term{
        public:
        Term() = default;
        Term(const Term&) = default;
        Term(Term&&) = default;
        Term& operator=(const Term&) = default;
        Term& operator=(Term&&) = default;
        ~Term() = default;

        void evaluate(Field& target_field);
    }; 

    template<typename derived>
    struct Expression{
        public:
        Expression() = default;
        Expression(const Expression&) = default;
        Expression(Expression&&) = default;
        Expression& operator=(const Expression&) = default;
        Expression& operator=(Expression&&) = default;
        ~Expression() = default;

        Expression(Term t0, Term t1) : term_0(t0), term_1(t1) {};


        void evaluate(Field& target_field) {
            static_cast<const derived&>(*this).evaluate(target_field);
        };


        const Term& term_0; 
        const Term& term_1;
    };

    struct Equation{
        public:
        Equation() = default;
        Equation(const Equation&) = default;
        Equation(Equation&&) = default;
        Equation& operator=(const Equation&) = default;
        Equation& operator=(Equation&&) = default;
        ~Equation() = default;

        Equation(const Expression& expr) : _expr(expr) {
            
            _expr.evaluate(target_field);
        }; // probably needs variadic template
        
        const Expression& _expr;
    };

    namespace fv
    {
        struct grad : public Term  // where this sit? -> multiple disc schemes may be an issue?
        {

            public:
                grad() = delete;
                grad(const grad&) = default;
                grad(grad&&) = default;
                grad& operator=(const grad&) = default;
                grad& operator=(grad&&) = default;
                ~grad() = default;

                grad(const Field&);

                void evaluate(Field& target_field) {

                };
        };

        struct div : public Term  // where this sit? -> multiple disc schemes may be an issue?
        {

            public:
                div() = delete;
                div(const div&) = default;
                div(div&&) = default;
                div& operator=(const div&) = default;
                div& operator=(div&&) = default;
                ~div() = default;

                div(const Field& phi, const Field& phi);

                void evaluate(Field& target_field) {

                };
        };

        struct ddt : public Term  // where this sit? -> multiple disc schemes may be an issue?
        {

            public:
                ddt() = delete;
                ddt(const ddt&) = default;
                ddt(grad&&) = default;
                ddt& operator=(const ddt&) = default;
                ddt& operator=(ddt&&) = default;
                ~grad() = default;

                ddt(Field);

                void evaluate(Field& target_field) {

                };
        };

        
    } // namespace fvOperators

    Expression operator+(const Term& t0, const Term& t1)
    {
        return Expression(t0, t1);
    }


    Expression operator==(const Term& t0, const Term& t1)
    {
        // do something
    }
 
} // namespace NeoFOAM

int main()
{ 
    using namespace NeoFOAM;
    Field p;
    Field alpha;

    Equation eq(
        fv::ddt(p) = fv::grad(p) + fv::grad(alpha)
        );

    return 0;
}