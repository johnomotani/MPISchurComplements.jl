using Combinatorics
using LinearAlgebra
using MPI
using Primes

include("julia-time-lu.jl")

const matrix_sizes = [128, 143, 256, 263, 512, 1024, 2048, 2057, 4096, 8192, 16384, 16397]
const dbr_values = (nothing, 1)
const nrhs = 10
const nmat_repeats_max = 10
const nrhs_repeats = 10
const tile_sizes = [32, 64, 128, 256, 512, 1024, 2048]

const logfile = "timings-julia.log"

function main()
    MPI.Init()
    nproc = MPI.Comm_size(MPI.COMM_WORLD)

    # Ensure BLAS only uses 1 thread, to avoid oversubscribing processes as we are already
    # fully parallelised.
    BLAS.set_num_threads(1)

    for n_shared ∈ reverse([prod(x) for x ∈ unique(combinations(factor(Vector, nproc)))]), distributed_block_rows ∈ dbr_values
        for n ∈ matrix_sizes
            if nproc ≤ 3 && n > 4096
                continue
            elseif nproc ≤ 8 && n > 8192
                continue
            end

            if n > 8192
                imat_max = 3
            elseif n > 1024
                imat_max = 5
            else
                imat_max = 10
            end

            if n > 4196
                nmat_repeats = min(3, nmat_repeats_max)
            elseif n > 1024
                nmat_repeats = min(5, nmat_repeats_max)
            else
                nmat_repeats = nmat_repeats_max
            end

            for imat ∈ 1:imat_max
                time_lu(ARGS[1], n_shared, distributed_block_rows, n, imat, nrhs,
                        nmat_repeats, nrhs_repeats, tile_sizes, logfile)
            end
        end
    end

    MPI.Finalize()
    return nothing
end

main()
