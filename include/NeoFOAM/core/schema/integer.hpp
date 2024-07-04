// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include <nlohmann/json.hpp>
#include <string>
#include <iostream>
#include <optional>

namespace NeoFOAM
{

struct Description
{
    std::string value;
};

struct MultipleOf
{
    double value;
};

struct Minimum
{
    double value;
};

struct Maximum
{
    double value;
};

struct ExclusiveMinimum
{
    double value;
};

struct ExclusiveMaximum
{
    double value;
};

class Number
{
public:

    std::optional<std::string> description;
    std::optional<double> multipleOf;
    std::optional<double> minimum;
    std::optional<double> maximum;
    std::optional<double> exclusiveMinimum;
    std::optional<double> exclusiveMaximum;

    Number() = default;

    template<typename... Params>
    Number(Params... params)
    {
        (set(params), ...); // Fold expression to call set method for each parameter
    }

    void set(const Description& desc);

    void set(const MultipleOf& mult);

    void set(const Minimum& min);

    void set(const Maximum& max);

    void set(const ExclusiveMinimum& exclMin);

    void set(const ExclusiveMaximum& exclMax);
};


}

namespace nlohmann
{
void to_json(nlohmann::json& j, const NeoFOAM::Number& schema);
}
