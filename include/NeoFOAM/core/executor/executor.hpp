// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/core/executor/CPUExecutor.hpp"
#include "NeoFOAM/core/executor/GPUExecutor.hpp"
#include "NeoFOAM/core/executor/OMPExecutor.hpp"
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
[[nodiscard]] constexpr bool operator==(const executor& lhs, const executor& rhs)
{
    return lhs.index() == rhs.index();
};

/**
 * @brief Checks if two executors are not equal, i.e. they are not of the same type.
 * @param lhs The first executor.
 * @param rhs The second executor.
 * @return True if the executors not are equal, false otherwise.
 */
[[nodiscard]] constexpr bool operator!=(const executor& lhs, const executor& rhs)
{
    return !(lhs == rhs);
};

} // namespace NeoFOAM
