// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/core/executor/CPUExecutor.hpp"
#include "NeoFOAM/core/executor/GPUExecutor.hpp"
#include "NeoFOAM/core/executor/OMPExecutor.hpp"
#include "NeoFOAM/core/stl_extention/variant.hpp"
#include <variant>

namespace NeoFOAM
{

using executor = std::variant<OMPExecutor, GPUExecutor, CPUExecutor>;

/**
 * @brief Checks if two executors are equal, i.e. they are of the same type.
 * @param lhs The first executor.
 * @param rhs The second executor.
 * @return True if the executors are equal, false otherwise.
 */
[[nodiscard]] bool operator==(const executor& lhs, const executor& rhs)
{
    return std::visit(overload {[](const OMPExecutor& lhs, const OMPExecutor& rhs)
                                { return lhs.exec_instance == rhs.exec_instance; },
                                [](const GPUExecutor& lhs, const GPUExecutor& rhs)
                                { return lhs.exec_instance == rhs.exec_instance; },
                                [](const CPUExecutor& lhs, const CPUExecutor& rhs)
                                { return lhs.exec_instance == rhs.exec_instance; },
                                [](const auto& lhs, const auto& rhs)
                                { return false; }},
                      lhs,
                      rhs);
};

/**
 * @brief Checks if two executors are not equal, i.e. they are not of the same type.
 * @param lhs The first executor.
 * @param rhs The second executor.
 * @return True if the executors not are equal, false otherwise.
 */
[[nodiscard]] bool operator!=(const executor& lhs, const executor& rhs)
{
    return !(lhs == rhs);
};

} // namespace NeoFOAM
