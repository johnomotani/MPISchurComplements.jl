#!/bin/bash

# Get path to BLAS and LAPACK and name of BLAS library from Julia, to pass to fortran compiler.
JULIA_BLAS_LIB=$(julia --project -O3 --startup-file=no -e 'using LinearAlgebra; println(LinearAlgebra.BLAS.lbt_get_config().loaded_libs[1].libname);')
JULIA_BLAS_DIR=$(dirname $JULIA_BLAS_LIB)
JULIA_BLAS_NAME=$(basename $JULIA_BLAS_LIB .so)

#Remove first 3 characters ('lib') to pass to linker
JULIA_BLAS_NAME=${JULIA_BLAS_NAME:3}

echo JULIA_BLAS_LIB=$JULIA_BLAS_LIB
echo JULIA_BLAS_DIR=$JULIA_BLAS_DIR
echo JULIA_BLAS_NAME=$JULIA_BLAS_NAME

# Compile fortran benchmark
# Use the `-Wl,-rpath` so the path to the library gets saved in the executable and we don't have to set $LD_LIBRARY_PATH.
h5pfc -O3 -o fortran-benchmark fortran-benchmark.f90 -lscalapack-openmpi -L$JULIA_BLAS_DIR -Wl,-rpath=$JULIA_BLAS_DIR -llapack -l$JULIA_BLAS_NAME


# Compile Julia benchmark
julia --project -O3 compile-julia-benchmark.jl
