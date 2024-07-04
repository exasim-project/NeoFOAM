// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include "NeoFOAM/core/schema/Number.hpp"

void NeoFOAM::Number::set(const Description& desc) { description = desc.value; }

void NeoFOAM::Number::set(const MultipleOf& mult) { multipleOf = mult.value; }

void NeoFOAM::Number::set(const Minimum& min) { minimum = min.value; }

void NeoFOAM::Number::set(const Maximum& max) { maximum = max.value; }

void NeoFOAM::Number::set(const ExclusiveMinimum& exclMin) { exclusiveMinimum = exclMin.value; }

void NeoFOAM::Number::set(const ExclusiveMaximum& exclMax) { exclusiveMaximum = exclMax.value; }

namespace nlohmann
{
void to_json(nlohmann::json& j, const NeoFOAM::Number& schema)
{
    j = nlohmann::json {{"type", "number"}};
    if (schema.description)
    {
        j["description"] = *schema.description;
    }
    if (schema.multipleOf)
    {
        j["multipleOf"] = *schema.multipleOf;
    }
    if (schema.minimum)
    {
        j["minimum"] = *schema.minimum;
    }
    if (schema.maximum)
    {
        j["maximum"] = *schema.maximum;
    }
    if (schema.exclusiveMinimum)
    {
        j["exclusiveMinimum"] = *schema.exclusiveMinimum;
    }
    if (schema.exclusiveMaximum)
    {
        j["exclusiveMaximum"] = *schema.exclusiveMaximum;
    }
}
}
