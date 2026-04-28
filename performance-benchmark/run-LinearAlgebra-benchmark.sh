#!/bin/bash

julia --project -O3 julia-LinearAlgebra-benchmark.jl matrices-and-rhs.h5 2>&1 | tee -a log-LinearAlgebra-benchmark.log
