# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: 2026 NeoFOAM authors

"""Pytest CLI options for the cylinder benchmark (``test_cylinder_runtime``).

Replaces the old ``NEOFOAM_BENCH_*`` environment variables: each option, when
given, overrides the matching typed constant on the test module before the tests
run. Defaults live in the test module — this only wires up per-run overrides.

    pytest benchmarks/test_cylinder_runtime.py -s --bench-rtol 1e-6 \
        --bench-targets 100000,800000 --bench-steps 6
"""

from __future__ import annotations

from typing import Any

import pytest

# (option flag, module attribute, converter). Comma-lists convert to typed lists.
_INT = int
_STR = str


def _ints(text: str) -> list[int]:
    return [int(float(t)) for t in text.split(",") if t.strip()]


def _floats(text: str) -> list[float]:
    return [float(t) for t in text.split(",") if t.strip()]


_OPTIONS: list[tuple[str, str, Any]] = [
    ("--bench-warmup", "N_WARMUP", _INT),
    ("--bench-steps", "N_TIMED", _INT),
    ("--bench-targets", "TARGETS", _ints),
    ("--bench-dx", "DX_LEVELS", _floats),
    ("--bench-maxsize-nx", "MAXSIZE_NX", _INT),
    ("--bench-maxsize", "MAXSIZE_LEVELS", _ints),
    ("--bench-maxsize-rtol", "MAXSIZE_RTOL", _STR),
    ("--bench-rtol", "BENCH_RTOL", _STR),
    ("--bench-csv-dir", "RESULTS_DIR", _STR),
]


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("cylinder-benchmark")
    for flag, attr, _conv in _OPTIONS:
        group.addoption(flag, default=None, help=f"override {attr}")
    group.addoption(
        "--bench-csv-append",
        action="store_true",
        default=False,
        help="append rows to existing CSVs instead of truncating (chunked runs)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Apply given options onto the test module's constants before collection."""
    from pathlib import Path

    import test_cylinder_runtime as mod

    for flag, attr, conv in _OPTIONS:
        raw = config.getoption(flag)
        if raw is None:
            continue
        value = Path(raw) if attr == "RESULTS_DIR" else conv(raw)
        setattr(mod, attr, value)
    if config.getoption("--bench-csv-append"):
        mod.CSV_APPEND = True
