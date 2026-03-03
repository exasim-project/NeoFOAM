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
  echo "  $0 <debug|profile> <tool-name> [--log <filename>] <application> [args...]"
    echo "---------------------------------------------------------------------------"
    echo "Enter debug or profile to select the mode, then the name of the tool to use."
    echo ""
    echo "Available tools for debug mode (used with the CMake preset "develop"):"
    echo "  kernel-logger"
    echo ""
    echo "Available tools for profile mode (used with the CMake preset "profiling"):"
    echo "  simple-kernel-timer"
    echo "  space-time-stack"
    echo "  memory-high-water-mark"
    exit 1
fi

MODE=$1
TOOL_NAME=$2
shift 2

# Optional log filename: allow `--log <file>` before the application command
LOGFILE=""
if [ "$1" = "--log" ]; then
  if [ -z "$2" ]; then
    echo "Error: --log requires a filename"
    exit 1
  fi
  LOGFILE="$2"
  shift 2
fi

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
echo "Running: $@"
echo ""

# ---------------------------------------------------------------------------
# Execute
# - If tool is not simple-kernel-timer: write stdout/stderr to a tool-specific
#   log file
# - If tool is simple-kernel-timer: ensure the .so directory is on PATH and
#   after the executable finishes try to run `kp_reader` on the generated .dat
#   file.
# ---------------------------------------------------------------------------

if [ "$TOOL_NAME" != "simple-kernel-timer" ]; then
  LOGFILE="${LOGFILE:-${TOOL_NAME}}.log"
  echo "Writing program output to $LOGFILE"
  "$@" >"$LOGFILE" 2>&1
else
  # Ensure the directory containing the simple-kernel-timer .so is on PATH
  # so helper executables can be found.
  if [ -n "$KOKKOS_TOOLS_LIBS" ]; then
    LIB_DIR="$(dirname "$KOKKOS_TOOLS_LIBS")"
    export PATH="$PATH:$LIB_DIR"
  fi

  # Run the executable (preserve exit status)
  "$@"
  EX_CODE=$?

  # Try to find kp_reader
  KP_READER="$(command -v kp_reader || true)"

  if [ -n "$KP_READER" ] && [ -x "$KP_READER" ]; then
    # Find the most recent .dat file in the current directory
    DATFILE="$(ls -t *.dat 2>/dev/null | head -n1 || true)"
    if [ -z "$DATFILE" ]; then
      echo "No .dat file produced; skipping kp_reader"
    else
      echo "Running kp_reader on $DATFILE"
      if [ -n "$LOGFILE" ]; then
        READER_LOG="${LOGFILE}.log"
      else
        READER_LOG="${DATFILE}.log"
      fi
      "$KP_READER" "$DATFILE" > "$READER_LOG" 2>&1 || true
      echo "Writing timing result to $READER_LOG"
    fi
  else
    echo "kp_reader not found; skipping post-processing"
  fi

  exit $EX_CODE
fi
