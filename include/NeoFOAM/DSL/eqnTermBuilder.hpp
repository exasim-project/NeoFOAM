// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023-2024 NeoFOAM authors
#pragma once

#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>

#include "NeoFOAM/core/primitives/scalar.hpp"
#include "NeoFOAM/DSL/eqnTerm.hpp"
#include "NeoFOAM/core/inputs.hpp"
#include <functional>

namespace fvcc = NeoFOAM::finiteVolume::cellCentred;

namespace NeoFOAM::DSL
{


// template<typename ValueType>
// class EqnTerm {};

template<typename ValueType>
class eqnTermBuilder
{
    using buildFunction = std::function<EqnTerm<ValueType>(const NeoFOAM::Input&)>;

public:

    eqnTermBuilder() = default;

    // template<typename T>
    // EqnTerm<ValueType> build(const NeoFOAM::Input& input)
    // {
    //     for (auto& buildFunction : buildFunctions_)
    //     {
    //         return buildFunction(input);
    //     }
    //     throw NeoFOAM::Error("No build function found for input");
    // }

    void push_back(buildFunction func) { buildFunctions_.push_back(func); }

    void push_back(eqnTermBuilder<ValueType> builder)
    {
        for (auto& buildFunction : builder)
        {
            buildFunctions_.push_back(buildFunction);
        }
    }

    // Iterator class
    class iterator
    {
    public:

        using value_type = buildFunction;

        iterator(typename std::vector<buildFunction>::iterator it) : it_(it) {}

        value_type& operator*() const { return *it_; }

        iterator& operator++()
        {
            ++it_;
            return *this;
        }

        bool operator!=(const iterator& other) const { return it_ != other.it_; }

    private:

        typename std::vector<buildFunction>::iterator it_;
    };

    iterator begin() { return iterator(buildFunctions_.begin()); }

    iterator end() { return iterator(buildFunctions_.end()); }

private:

    std::vector<buildFunction> buildFunctions_;
};

template<typename ValueType>
eqnTermBuilder<ValueType> operator+(eqnTermBuilder<ValueType> lhs, eqnTermBuilder<ValueType> rhs)
{
    lhs.push_back(rhs);
    return lhs;
}

template<typename ValueType>
eqnTermBuilder<ValueType> operator-(eqnTermBuilder<ValueType> lhs, eqnTermBuilder<ValueType> rhs)
{
    lhs.push_back(rhs);
    return lhs;
}

} // namespace NeoFOAM::DSL
