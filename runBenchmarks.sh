#!/bin/bash

./build/src/benchmarks/blas/bench_blas -r XML > blas.xml
python plotBenchmarks.py

