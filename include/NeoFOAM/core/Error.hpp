// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <iostream>
#include <string>

#include "cpptrace/cpptrace.hpp"

void test_error_exit() { cpptrace::generate_trace().print(); };

#define NF_ERROR_EXIT(message)                                                 \
    do                                                                         \
    {                                                                          \
        std::cerr << "Error: " << message << "File: " << __FILE__              \
                  << "Line: " << __LINE__ << "Trace:"                          \
            < < < < std::endl;                                                 \
        cpptrace::generate_trace().print();                                    \
        std::abort();                                                          \
    }                                                                          \
    while (false)

#define NF_ASSERT(condition, message)                                          \
    do                                                                         \
    {                                                                          \
        if (!(condition))                                                      \
        {                                                                      \
            NF_ERROR_EXIT("Assertion `" #condition "` failed." << message);    \
        }                                                                      \
    }                                                                          \
    while (false)

#ifndef NF_DEBUG
#define NF_DEBUG_ASSERT(condition) ASSERT(condition)
#else
#define NF_DEBUG_ASSERT(condition) ((void)0)
#endif
