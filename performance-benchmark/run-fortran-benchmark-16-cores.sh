#!/bin/bash

# For each number of MPI ranks, use all possible numbers of BLAS threads while
# staying within 32 physical cores.
for NT in {1..16}; do
  OPENBLAS_NUM_THREADS=$NT mpirun -np 1 ./fortran-benchmark matrices-and-rhs.h5
done
for NT in {1..8}; do
  OPENBLAS_NUM_THREADS=$NT mpirun -np 2 ./fortran-benchmark matrices-and-rhs.h5
done
for NT in {1..5}; do
  OPENBLAS_NUM_THREADS=$NT mpirun -np 3 ./fortran-benchmark matrices-and-rhs.h5
done
for NT in {1..4}; do
  OPENBLAS_NUM_THREADS=$NT mpirun -np 4 ./fortran-benchmark matrices-and-rhs.h5
done
for NT in {1..3}; do
  OPENBLAS_NUM_THREADS=$NT mpirun -np 5 ./fortran-benchmark matrices-and-rhs.h5
done
for NT in {1..2}; do
  for NP in {6..8} do
    OPENBLAS_NUM_THREADS=$NT mpirun -np $NP ./fortran-benchmark matrices-and-rhs.h5
  done
done
for NP in {9..16} do
  OPENBLAS_NUM_THREADS=1 mpirun -np $NP ./fortran-benchmark matrices-and-rhs.h5
done
