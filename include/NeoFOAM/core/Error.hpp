// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <iostream>
#include <string>

namespace NeoFOAM
{

class Error
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

    Error(std::string) {};
};

} // namespace NeoFOAM
