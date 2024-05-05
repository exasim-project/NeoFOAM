// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include "NeoFOAM/core/executor/CPUExecutor.hpp"
#include "NeoFOAM/core/executor/GPUExecutor.hpp"
#include "NeoFOAM/core/executor/OMPExecutor.hpp"
#include <variant>

namespace NeoFOAM
{

using Executor = std::variant<OMPExecutor, GPUExecutor, CPUExecutor>;

/**
 * @brief Checks if two executors are equal, i.e. they are of the same type.
 * @param lhs The first executor.
 * @param rhs The second executor.
 * @return True if the executors are equal, false otherwise.
 */
struct Visitor
{
    template<typename ExecLhs, typename ExecRhs>
    [[nodiscard]] inline bool operator()([[maybe_unused]] const ExecLhs&, [[maybe_unused]] const ExecRhs&) const
    {
        if constexpr (std::is_same_v<ExecLhs, ExecRhs>)
        {
            return typename ExecLhs::exec() == typename ExecRhs::exec();
        }
        else
        {
            return false;
        }
    }
};

[[nodiscard]] inline bool operator==(const Executor& lhs, const Executor& rhs)
{
    return std::visit(Visitor {}, lhs, rhs);
};

/**
 * @brief Checks if two executors are not equal, i.e. they are not of the same type.
 * @param lhs The first executor.
 * @param rhs The second executor.
 * @return True if the executors not are equal, false otherwise.
 */
[[nodiscard]] inline bool operator!=(const Executor& lhs, const Executor& rhs)
{
    return !(lhs == rhs);
};

} // namespace NeoFOAM
