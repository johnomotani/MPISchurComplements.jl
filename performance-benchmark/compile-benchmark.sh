#!/bin/bash

h5pfc -O3 -o fortran-benchmark fortran-benchmark.f90 -lscalapack-openmpi -llapack -lblas
