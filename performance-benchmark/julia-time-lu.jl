using Dates
using HDF5
using LinearAlgebra
using MPI
using MPISchurComplements
using MPISchurComplements.DenseLUs

const logfile = "timings-julia.log"
const nrhs = 10
const nmat_repeats = 10
const nrhs_repeats = 10
const tile_sizes = [32, 64, 128, 256, 512, 1024, 2048]

include("../test/utils.jl")

"""
    time_lu(filename, n_shared, n, imat)

Time the LU factorization and solve for a matrix of size `n` and label `imat` loaded from
the HDF5 file called `filename`, running with shared-memory blocks of size `n_shared`
processes.
"""
function time_lu(filename, n_shared, n, imat)
    comm_world = MPI.COMM_WORLD
    comm, distributed_comm, distributed_nproc, distributed_rank, shared_comm,
        shared_nproc, shared_rank, allocate_shared_float, allocate_shared_int,
        local_win_store_float, local_win_store_int = get_comms(n_shared, true)

    A = allocate_shared_float(n, n)
    rhs_array = allocate_shared_float(n, nrhs)
    x = allocate_shared_float(n)
    if distributed_rank == 0 && shared_rank == 0
        file = h5open(filename, "r")
        mat_name = "matrix-$n-$imat"
        A .= read(file[mat_name])
        vec_name = "rhs-$n"
        rhs_array .= read(file[vec_name])
        close(file)
    end

    for nb ∈ tile_sizes
        if nb > n
            continue
        end
        for irepeatmat ∈ 1:nmat_repeats
            MPI.Barrier(comm_world)
            t0 = time_ns()
            A_lu = dense_lu(A, nb, comm, shared_comm, distributed_comm, allocate_shared_float,
                            allocate_shared_int; skip_factorization=true, check_lu=false)
            MPI.Barrier(comm_world)
            t1 = time_ns()
            t_setup = (t1 - t0) / 1e9

            if distributed_rank == 0 && shared_rank == 0
                println(now())
                println("benchmark: matrix=$mat_name  rhs=$vec_name  file='$filename'")
                println("  nproc = $(distributed_nproc * shared_nproc)  n_shared=$n_shared  n = $n  nb = $nb  nrhs = $nrhs")
            end

            t_trisolve_array = fill(Float64(Inf), nrhs)

            # Time the factorization
            MPI.Barrier(comm_world)
            t0 = time_ns()
            lu!(A_lu, A)
            MPI.Barrier(comm_world)
            t1 = time_ns()
            t_factorisation = (t1 - t0) / 1e9

            for i ∈ 1:nrhs_repeats
                for (irhs, rhs) ∈ enumerate(eachcol(rhs_array))
                    MPI.Barrier(comm_world)
                    t0 = time_ns()
                    ldiv!(x, A_lu, rhs)
                    t1 = time_ns()
                    t_trisolve = (t1 - t0) / 1e9
                    t_trisolve_array[irhs] = min(t_trisolve_array[irhs], t_trisolve)
                end
            end

            # Print and save timing summary.
            if distributed_rank == 0 && shared_rank == 0
                t_trisolve_min = minimum(t_trisolve_array)
                t_trisolve_max = maximum(t_trisolve_array)
                t_trisolve_mean = sum(t_trisolve_array) / length(t_trisolve_array)

                println("Timing summary (wall-clock):")
                println("  DenseLU setup: $t_setup s")
                println("  LU factorisation: $t_factorisation s")
                println("  Triangular solve:")
                println("    min  : $t_trisolve_min s")
                println("    mean  : $t_trisolve_mean s")
                println("    max  : $t_trisolve_max s")
                println("  Total (factorisation + mean solve) : $(t_factorisation + t_trisolve_mean) s")
                println()

                open(logfile, "a") do io
                    println(io, "$n $nb $imat $distributed_nproc $shared_nproc $t_factorisation $t_trisolve_min $t_trisolve_mean $t_trisolve_max $(t_factorisation + t_trisolve_mean)")
                end
            end

            # Delete all shared arrays except the first 3, which are A, rhs_array, and x.
            if local_win_store_float !== nothing
                # Free the MPI.Win objects, because if they are free'd by the garbage collector
                # it may cause an MPI error or hang.
                for w ∈ local_win_store_float[4:end]
                    MPI.free(w)
                end
            end
            if local_win_store_int !== nothing
                # Free the MPI.Win objects, because if they are free'd by the garbage collector
                # it may cause an MPI error or hang.
                for w ∈ local_win_store_int[4:end]
                    MPI.free(w)
                end
            end
        end
    end

    if local_win_store_float !== nothing
        # Free the MPI.Win objects, because if they are free'd by the garbage collector
        # it may cause an MPI error or hang.
        for w ∈ local_win_store_float
            MPI.free(w)
        end
        resize!(local_win_store_float, 0)
    end
    if local_win_store_int !== nothing
        # Free the MPI.Win objects, because if they are free'd by the garbage collector
        # it may cause an MPI error or hang.
        for w ∈ local_win_store_int
            MPI.free(w)
        end
        resize!(local_win_store_int, 0)
    end

    MPI.Barrier(MPI.COMM_WORLD)

    return nothing
end
