// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <iostream>
#include <string>

namespace NeoFOAM
{

class error
{
public:

    /**
     * @brief Exit the program with an error message.
     *
     * @param errNo The error number to exit with.
     * @param location Default argument for the location of the error.
     */
    void exit(const int errNo = 1)
    {
        std::cout << "Error: " << errNo << '\n';
        std::exit(errNo);
    };

    error(std::string) {};
};

extern error FatalError;
} // namespace NeoFOAM
