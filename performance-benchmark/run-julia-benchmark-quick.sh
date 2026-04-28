#!/bin/bash

for NP in 1 2 4 6 8 12 16 24 32 48 64 96 128 192 256; do
  if [ "$NP" -gt "$1" ]; then
    exit 0
  fi
  mpirun -np $NP julia --project -O3 -J julia-benchmark.so julia-benchmark-quick.jl matrices-and-rhs.h5 2>&1 | tee -a log-julia-benchmark-quick.log
done
