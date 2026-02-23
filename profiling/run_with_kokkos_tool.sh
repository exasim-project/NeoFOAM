#!/usr/bin/env bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
NEOFOAM_SRC_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
BUILD_DIR="${NEOFOAM_SRC_DIR}/build"

TOOL_NAME=$1
shift

if [ -z "$TOOL_NAME" ]; then
    echo "Usage:"
    echo "  $0 <tool-name>"
    echo ""
    echo "Available tools:"
    echo "  simple kernel timer"
    echo "  space time stack"
    echo "  memory usage"
    exit 1
fi

TOOLS_DIR="${BUILD_DIR}/profiling/kokkos_tools_build/profiling"

case "$TOOL_NAME" in
  simple_kernel_timer)
    export KOKKOS_TOOLS_LIBS=${TOOLS_DIR}/simple-kernel-timer/libkp_kernel_timer.so
    export PATH=${TOOLS_DIR}/simple-kernel-timer:$PATH
    ;;
  space_time_stack)
    export KOKKOS_TOOLS_LIBS=${TOOLS_DIR}/space-time-stack/libkp_space_time_stack.so
    ;;
  memory_usage)
    export KOKKOS_TOOLS_LIBS=${TOOLS_DIR}/memory-usage/libkp_memory_usage.so
    ;;
  *)
    echo "Unknown tool: $TOOL_NAME"
    exit 1
    ;;
esac

echo "Using tool: $TOOL_NAME"
echo "KOKKOS_TOOLS_LIBS=$KOKKOS_TOOLS_LIBS"
echo "-----------------------------------"

