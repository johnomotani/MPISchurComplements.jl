#!/bin/bash

# For each number of MPI ranks, use all possible numbers of BLAS threads while
# staying within 32 physical cores.
#for NT in {1..32}; do
#  OPENBLAS_NUM_THREADS=$NT mpirun -np 1 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
#done
#for NT in {1..16}; do
#  OPENBLAS_NUM_THREADS=$NT mpirun -np 2 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
#done
#for NT in {1..10}; do
#  OPENBLAS_NUM_THREADS=$NT mpirun -np 3 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
#done
#for NT in {1..8}; do
#  OPENBLAS_NUM_THREADS=$NT mpirun -np 4 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
#done
#for NT in {1..6}; do
#  OPENBLAS_NUM_THREADS=$NT mpirun -np 5 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
#done
#for NT in {1..5}; do
#  OPENBLAS_NUM_THREADS=$NT mpirun -np 6 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
#done
#for NT in {1..4}; do
#  OPENBLAS_NUM_THREADS=$NT mpirun -np 7 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
#done
#for NT in {1..4}; do
#  OPENBLAS_NUM_THREADS=$NT mpirun -np 8 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
#done
#for NT in {1..3}; do
#  OPENBLAS_NUM_THREADS=$NT mpirun -np 9 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
#done
#for NT in {1..3}; do
#  OPENBLAS_NUM_THREADS=$NT mpirun -np 10 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
#done
for NT in {1..2}; do
  #for NP in {11..16}; do
  for NP in {1..16}; do
    OPENBLAS_NUM_THREADS=$NT mpirun -np $NP ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
  done
done
for NP in {17..32}; do
  OPENBLAS_NUM_THREADS=1 mpirun -np $NP ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
done
