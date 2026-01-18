#!/bin/bash
cd "${0%/*}" || exit

current_dir=$(pwd)

# run benchmarks
run_benchmark() {
    echo "Running $2 benchmarks in: $1"
    # mkdir -p $1/results
    find $1 -name "Allrun" -exec {} bench_$2 \;

    echo "Gathering results..." $1
    python gatherResults.py $1
    python plot_results.py $1

    echo "Completed benchmark: $1"
}

# Execute the benchmark commands
# echo "Creating study..."
# python createStudies.py

# Define benchmarks to run
# benchmarks=("explicitOperators" "implicitOperators" "dsl")
# # Run each benchmark
# for benchmark in "${benchmarks[@]}"; do
#     run_benchmark "$current_dir/$benchmark" $benchmark
# done

find "$current_dir/dsl" -name "Allclean" -exec {} ./Allclean \;
run_benchmark "$current_dir/dsl" "dsl"

find "$current_dir/implicitOperators" -name "Allclean" -exec {} ./Allclean \;
run_benchmark "$current_dir/implicitOperators" "implicitOperators"

find "$current_dir/explicitOperators" -name "Allclean" -exec {} ./Allclean \;
run_benchmark "$current_dir/explicitOperators" "explicitOperators"
