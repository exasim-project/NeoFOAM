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

} // namespace NeoFOAM
