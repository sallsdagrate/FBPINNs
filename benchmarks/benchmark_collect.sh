#!/usr/bin/env bash
set -euo pipefail

# check that exactly two arguments were provided
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <problem> <benchmarkname>"
  exit 1
fi

problem="$1"
benchmark="$2"

# base folder for this problem
base="benchmarks/runs/${problem}"
# final destination for this benchmark
dest="${base}/${benchmark}"

# ensure the problem folder exists
mkdir -p "$base"
# ensure the benchmark folder exists
mkdir -p "$dest"

mv -- results/saved_arrays/*.json results/saved_arrays/*.npy results/saved_arrays/*.npz results/summaries/test/*.png "$dest"/

echo "Benchmark files moved → $dest"
