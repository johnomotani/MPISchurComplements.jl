#!/bin/bash

for NP in $(seq 1 $1); do
  mpirun -np $NP julia --project -O3 -J julia-benchmark.so julia-benchmark.jl matrices-and-rhs.h5 2>&1 | tee -a log-julia-benchmark.log
done
