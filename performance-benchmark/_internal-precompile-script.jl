using MPI

include("julia-time-lu.jl")

MPI.Init()
time_lu("matrices-and-rhs.h5", 1, 128, 1)
MPI.Finalize()
