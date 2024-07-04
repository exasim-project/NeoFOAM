// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 NeoFOAM authors

#include <nlohmann/json.hpp>
#include <string>
#include <iostream>
#include <optional>

namespace NeoFOAM
{
namespace schema
{

    struct Title
    {
        std::string value;
    };

    struct Description
    {
        std::string value;
    };


    struct Default
    {
        double value;
    };

    struct Example
    {
        std::string value;
    };


    class Annotations
    {
    public:

        std::string title;
        std::string description;
        std::optional<double> defaultValues;
        std::optional<double> example;


        Annotations() = default;

        template<typename... Params>
        Annotations(Params... params)
        {
            (set(params), ...); // Fold expression to call set method for each parameter
        }

        void set(const Title& title);

        void set(const Description& desc);

        void set(const Default& def);

        void set(const Example& ex);
    };


}
}

namespace nlohmann
{
void to_json(nlohmann::json& j, const NeoFOAM::schema::Annotations& schema);
}
