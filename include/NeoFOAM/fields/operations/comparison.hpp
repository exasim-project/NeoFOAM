// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <span>

#include "NeoFOAM/fields/field.hpp"

namespace NeoFOAM
{

template<typename T>
bool equal(Field<T>& field, T value)
{
    auto hostField = field.copyToHost();
    auto hostSpan = hostField.span();
    for (size_t i = 0; i < hostSpan.size(); i++)
    {
        if (hostSpan[i] != value)
        {
            return false;
        }
    }
    return true;
};

template<typename T>
bool equal(const Field<T>& field, const Field<T>& field2)
{
    auto [hostField, hostField2] = copyToHosts(field, field2);
    auto [hostSpan, hostSpan2] = spans(hostField, hostField2);

    if (hostSpan.size() != hostSpan2.size())
    {
        return false;
    }

    for (size_t i = 0; i < hostSpan.size(); i++)
    {
        if (hostSpan[i] != hostSpan2[i])
        {
            return false;
        }
    }

    return true;
};

template<typename T>
bool equal(const Field<T>& field, std::span<T> span2)
{
    auto hostSpan = field.copyToHost().span();

    if (hostSpan.size() != span2.size())
    {
        return false;
    }

    for (size_t i = 0; i < hostSpan.size(); i++)
    {
        if (hostSpan[i] != span2[i])
        {
            return false;
        }
    }

    return true;
}

} // namespace NeoFOAM
