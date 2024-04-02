// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors

#include <cxxopts.hpp>
#include <iostream>


#include "NeoFOAM/core/runTime.hpp"


void NeoFOAM::runTime::handle_cli_args(int argc, char* argv[], std::string program_name, std::string one_line_description)
{
    cxxopts::Options options(program_name, one_line_description);
    options.add_options()("d,debug", "Enable debugging") // a bool parameter
        ("h,help", "Print help")                         // a bool parameter
        ("c,case", "File name", cxxopts::value<std::string>())("v,version", "Print version and exit", cxxopts::value<bool>()->default_value("false"))("p,parallel", "Run in parallel", cxxopts::value<bool>()->default_value("false"));
    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (result.count("version"))
    {
        std::string version {"0.0.0"};
        std::string build {"aaaaa"};
        std::cout << "NeoFOAM version: " << version << " build: " << build << std::endl;
        exit(0);
    }
}
