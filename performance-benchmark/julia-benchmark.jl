using Combinatorics
using MPI
using Primes

include("julia-time-lu.jl")

const matrix_sizes = [128, 143, 256, 263, 512, 1024, 2048, 2057, 4096, 8192, 16384, 16397]

function main()
    MPI.Init()
    nproc = MPI.Comm_size(MPI.COMM_WORLD)

    for n_shared ∈ [prod(x) for x ∈ unique(combinations(factor(Vector, nproc)))]
        for n ∈ matrix_sizes
            for imat ∈ 1:10
                time_lu(ARGS[1], n_shared, n, imat)
            end
        end
    end

    MPI.Finalize()
    return nothing
end

main()
