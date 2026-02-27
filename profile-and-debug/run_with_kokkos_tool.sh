# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# SPDX-License-Identifier: Unlicense

#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NEOFOAM_SRC_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
BUILD_DIR="${NEOFOAM_SRC_DIR}/build"

if [ $# -lt 3 ]; then
    echo "Usage:"
    echo "  $0 <debug|profile> <tool-name> <application> [args...]"
    echo ""
    echo "Available debugging tools:"
    echo "  kernel logger"
    echo ""
    echo "Available profiling tools:"
    echo "  simple-kernel-timer"
    echo "  space-time-stack"
    echo "  memory-high-water-mark"
    exit 1
fi

MODE=$1
TOOL_NAME=$2
shift 2

# ---------------------------------------------------------------------------
# Select tools directory based on mode
# ---------------------------------------------------------------------------

case "$MODE" in
  debug)
    TOOLS_DIR="${BUILD_DIR}/develop/kokkos_tools_build/debugging"
    ;;
  profile)
    TOOLS_DIR="${BUILD_DIR}/profiling/kokkos_tools_build/profiling"
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Must be 'debug' or 'profile'"
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Select tool library
# ---------------------------------------------------------------------------

case "$TOOL_NAME" in
  kernel-logger)
    export KOKKOS_TOOLS_LIBS=${TOOLS_DIR}/kernel-logger/libkp_kernel_logger.so
    ;;
  simple-kernel-timer)
    export KOKKOS_TOOLS_LIBS=${TOOLS_DIR}/simple-kernel-timer/libkp_kernel_timer.so
    ;;
  space-time-stack)
    export KOKKOS_TOOLS_LIBS=${TOOLS_DIR}/space-time-stack/libkp_space_time_stack.so
    ;;
  memory-high-water-mark)
    export KOKKOS_TOOLS_LIBS=${TOOLS_DIR}/memory-hwm/libkp_hwm.so
    ;;
  *)
    echo "Unknown tool: $TOOL_NAME"
    exit 1
    ;;
esac

# ---------------------------------------------------------------------------
# Info
# ---------------------------------------------------------------------------

echo "Mode: $MODE"
echo "Using tool: $TOOL_NAME"
echo "KOKKOS_TOOLS_LIBS=$KOKKOS_TOOLS_LIBS"
echo "Running: $@"
echo ""

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------

exec "$@"
