// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <vector>
#include <any>
#include <string>
#include <iostream>

namespace NeoFOAM
{


class TokenList
{
public:

    TokenList() = default;

    TokenList(const std::vector<std::any>& data);

    TokenList(const std::initializer_list<std::any>& initList);

    void insert(const std::any& value);

    void remove(size_t index);

    [[nodiscard]] bool empty() const;

    [[nodiscard]] size_t size() const;


    template<typename T>
    [[nodiscard]] T& get(const size_t& idx)
    {
        return std::any_cast<T&>(data_.at(idx));
    }


    template<typename T>
    [[nodiscard]] const T& get(const size_t& idx) const
    {
        std::cout << "const T& get(const size_t& idx) const" << std::endl;
        return std::any_cast<const T&>(data_.at(idx));
    }

    [[nodiscard]] std::vector<std::any>& tokens();


private:

    std::vector<std::any> data_;
};

} // namespace NeoFOAM
