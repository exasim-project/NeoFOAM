// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include "NeoFOAM/mesh/stencil/StencilDataBase.hpp"

std::any& NeoFOAM::StencilDataBase::operator[](const std::string& key)
{
    return stencilDB_.at(key);
}

const std::any& NeoFOAM::StencilDataBase::operator[](const std::string& key) const
{
    return stencilDB_.at(key);
}

bool NeoFOAM::StencilDataBase::contains(const std::string& key) const
{
    return stencilDB_.find(key) != stencilDB_.end();
}
