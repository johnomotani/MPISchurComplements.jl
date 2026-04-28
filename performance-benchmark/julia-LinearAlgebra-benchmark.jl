using Dates
using HDF5
using LinearAlgebra

const logfile = "timings-linearalgebra.log"
const nrhs = 10
const nmat_repeats = 10
const nrhs_repeats = 10
const matrix_sizes = [128, 143, 256, 263, 512, 1024, 2048, 2057, 4096, 8192, 16384, 16397]
const nproc = 1

"""
    time_lu(filename, n_threads, n, imat)

Time the LU factorization and solve for a matrix of size `n` and label `imat` loaded from
the HDF5 file called `filename`, using `n_threads` threads in BLAS.
"""
function time_lu(filename, n_threads, n, imat)
    x = Vector{Float64}(undef,n)

    file = h5open(filename, "r")
    mat_name = "matrix-$n-$imat"
    A = read(file[mat_name])
    vec_name = "rhs-$n"
    rhs_array = read(file[vec_name])
    close(file)

    nb = n # We do not use 'tiles' for this serial version, so effectively 'tile size' is the matrix size.
    BLAS.set_num_threads(n_threads)

    A_lu_storage = similar(A)

    for irepeatmat ∈ 1:nmat_repeats
        A_lu_storage .= A

        println(now())
        println("benchmark: matrix=$mat_name  rhs=$vec_name  file='$filename'")
        println("  nproc = $(nproc)  n_threads=$n_threads  n = $n  nb = $nb  nrhs = $nrhs")

        t_trisolve_array = fill(Float64(Inf), nrhs)

        # Time the factorization
        t0 = time_ns()
        A_lu = lu!(A_lu_storage; check=false)
        t1 = time_ns()
        t_factorisation = (t1 - t0) / 1e9

        for i ∈ 1:nrhs_repeats
            for (irhs, rhs) ∈ enumerate(eachcol(rhs_array))
                t0 = time_ns()
                ldiv!(x, A_lu, rhs)
                t1 = time_ns()
                t_trisolve = (t1 - t0) / 1e9
                t_trisolve_array[irhs] = min(t_trisolve_array[irhs], t_trisolve)
            end
        end

        # Print and save timing summary.
        t_trisolve_min = minimum(t_trisolve_array)
        t_trisolve_max = maximum(t_trisolve_array)
        t_trisolve_mean = sum(t_trisolve_array) / length(t_trisolve_array)

        println("Timing summary (wall-clock):")
        println("  LU factorisation: $t_factorisation s")
        println("  Triangular solve:")
        println("    min  : $t_trisolve_min s")
        println("    mean  : $t_trisolve_mean s")
        println("    max  : $t_trisolve_max s")
        println("  Total (factorisation + mean solve) : $(t_factorisation + t_trisolve_mean) s")
        println()

        open(logfile, "a") do io
            println(io, "$n $nb $imat $nproc $n_threads 0 $t_factorisation $t_trisolve_min $t_trisolve_mean $t_trisolve_max $(t_factorisation + t_trisolve_mean)")
        end
    end

    return nothing
end

function main()
    max_threads = BLAS.get_num_threads()

    for n_threads ∈ 1:max_threads
        for n ∈ matrix_sizes
            for imat ∈ 1:10
                time_lu(ARGS[1], n_threads, n, imat)
            end
        end
    end

    return nothing
end

main()
