// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/fields/Field.hpp"
#include <span>

namespace NeoFOAM
{

template< typename T >
bool compare(Field<T>& field, T value)
{
    auto hostSpan = field.copyToHost().field();
    for(int i = 0; i < hostSpan.size(); i++)
    {
        if (hostSpan[i] != value)
        {
            return false;
        }
    }
    return true;
};

template< typename T >
bool compare(Field<T>& field, Field<T>& field2)
{
    auto hostSpan = field.copyToHost().field();
    auto hostSpan2 = field2.copyToHost().field();
    
    if (hostSpan.size() != hostSpan2.size())
    {
        return false;
    }
    
    for(int i = 0; i < hostSpan.size(); i++)
    {
        if (hostSpan[i] != hostSpan2[i])
        {
            return false;
        }
    }
    
    return true;
};

template< typename T >
bool compare(Field<T>& field, std::span<T> span2)
{
    auto hostSpan = field.copyToHost().field();
    
    if (hostSpan.size() != span2.size())
    {
        return false;
    }
    
    for(int i = 0; i < hostSpan.size(); i++)
    {
        if (hostSpan[i] != span2[i])
        {
            return false;
        }
    }
    
    return true;

}

} // namespace NeoFOAM