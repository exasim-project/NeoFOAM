#!/bin/bash
# SPDX-License-Identifier: Unlicense
# SPDX-FileCopyrightText: 2023 NeoFOAM authors

./build/src/benchmarks/blas/bench_blas -r XML > blas.xml
python plotBenchmarks.py

