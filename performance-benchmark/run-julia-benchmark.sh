#!/bin/bash

for NP in {1..$1} do
  mpirun -np $NP julia --project -O3 -J julia-benchmark.so julia-benchmark.jl matrices-and-rhs.h5
done
