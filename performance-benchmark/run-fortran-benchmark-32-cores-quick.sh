#!/bin/bash

OPENBLAS_NUM_THREADS=1 mpirun -np 1 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log

OPENBLAS_NUM_THREADS=1 mpirun -np 2 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
OPENBLAS_NUM_THREADS=2 mpirun -np 1 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log

OPENBLAS_NUM_THREADS=1 mpirun -np 4 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
OPENBLAS_NUM_THREADS=2 mpirun -np 2 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
OPENBLAS_NUM_THREADS=4 mpirun -np 1 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log

OPENBLAS_NUM_THREADS=1 mpirun -np 6 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
OPENBLAS_NUM_THREADS=2 mpirun -np 3 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
OPENBLAS_NUM_THREADS=3 mpirun -np 2 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log

OPENBLAS_NUM_THREADS=1 mpirun -np 8 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
OPENBLAS_NUM_THREADS=2 mpirun -np 4 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
OPENBLAS_NUM_THREADS=4 mpirun -np 2 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log

OPENBLAS_NUM_THREADS=1 mpirun -np 12 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
OPENBLAS_NUM_THREADS=2 mpirun -np 6 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
OPENBLAS_NUM_THREADS=3 mpirun -np 4 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log

OPENBLAS_NUM_THREADS=1 mpirun -np 16 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
OPENBLAS_NUM_THREADS=2 mpirun -np 8 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
OPENBLAS_NUM_THREADS=4 mpirun -np 4 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log

OPENBLAS_NUM_THREADS=1 mpirun -np 24 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
OPENBLAS_NUM_THREADS=2 mpirun -np 12 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
OPENBLAS_NUM_THREADS=3 mpirun -np 8 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log

OPENBLAS_NUM_THREADS=1 mpirun -np 24 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
OPENBLAS_NUM_THREADS=2 mpirun -np 12 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
OPENBLAS_NUM_THREADS=3 mpirun -np 8 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log

OPENBLAS_NUM_THREADS=1 mpirun -np 32 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
OPENBLAS_NUM_THREADS=2 mpirun -np 16 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
OPENBLAS_NUM_THREADS=4 mpirun -np 8 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
OPENBLAS_NUM_THREADS=8 mpirun -np 4 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
OPENBLAS_NUM_THREADS=16 mpirun -np 2 ./fortran-benchmark matrices-and-rhs.h5 2>&1 | tee -a log-fortran-benchmark.log
