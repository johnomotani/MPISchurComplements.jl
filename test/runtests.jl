using LinearAlgebra

# Ensure BLAS only uses 1 thread, to avoid oversubscribing processes as we are probably
# already fully parallelised.
BLAS.set_num_threads(1)

using MPI
using StableRNGs
using Test

using MPISchurComplements
using MPISchurComplements.FakeMPILUs

include("utils.jl")
include("simple_matrix.jl")
include("finite_element.jl")

function runtests()
    if !MPI.Initialized()
        MPI.Init()
    end
    @testset "MPISchurComplements" begin
        simple_matrix_tests()
        finite_element_tests()
    end
end
runtests()
