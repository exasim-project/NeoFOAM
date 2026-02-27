# SPDX-FileCopyrightText: 2026 NeoFOAM authors
#
# SPDX-License-Identifier: Unlicense

#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NEOFOAM_SRC_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
BUILD_DIR="${NEOFOAM_SRC_DIR}/build"

if [ $# -lt 1 ]; then
    echo "Usage:"
    echo "  $0 <tool-name>"
    echo ""
    echo "Available tools:"
    echo "  simple-kernel-timer"
    echo "  space-time-stack"
    echo "  memory-high-water-mark"
    exit 1
fi

TOOL_NAME=$1
shift

TOOLS_DIR="${BUILD_DIR}/profiling/kokkos_tools_build/profiling"

case "$TOOL_NAME" in
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

echo "Using tool: $TOOL_NAME"
echo "KOKKOS_TOOLS_LIBS=$KOKKOS_TOOLS_LIBS"

# If no command given, show usage and exit
if [ $# -eq 0 ]; then
  echo ""
  echo "No application specified to run with the tool."
  echo "Usage:"
  echo "  $0 <tool-name> <application> [args...]"
  echo "Example:"
  echo "  $0 simple-kernel-timer path-to-neoIcoFoam"
  exit 1
fi

echo "Running: $@"

# Execute the provided command with the tool environment
exec "$@"
