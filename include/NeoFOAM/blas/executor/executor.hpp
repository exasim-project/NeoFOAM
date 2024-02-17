// SPDX-License-Identifier: MPL-2.0
// SPDX-FileCopyrightText: 2023 NeoFOAM authors
#pragma once

#include <variant>
#include "NeoFOAM/blas/executor/ompExecutor.hpp"
#include "NeoFOAM/blas/executor/GPUExecutor.hpp"
#include "NeoFOAM/blas/executor/CPUExecutor.hpp"

namespace NeoFOAM
{

    using executor = std::variant<ompExecutor, GPUExecutor, CPUExecutor>;

} // namespace NeoFOAM
