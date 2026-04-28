#!/bin/bash

# Set up MPI and HDF5 for Julia
HDF5_DIR=""
while [[ -z $HDF5_DIR ]]; do
  echo "Enter path to directory containing HDF5 library:"
  read -e -p "> "  HDF5_DIR
  echo
done

julia --project -O3 -e "using MPIPreferences, HDF5; MPIPreferences.use_system_binary(); Sys.isapple() ? HDF5.API.set_libraries!(joinpath(\"$HDF5_DIR\", \"libhdf5.dylib\"), joinpath(\"$HDF5_DIR\", \"libhdf5_hl.dylib\")) : HDF5.API.set_libraries!(joinpath(\"$HDF5_DIR\", \"libhdf5.so\"), joinpath(\"$HDF5_DIR\", \"libhdf5_hl.so\"))"
