using Dates
using HDF5
using LinearAlgebra
using MPI
using TimerOutputs
using MPISchurComplements
using MPISchurComplements.DenseLUs
using TeeStreams

const imat = 1
const nrhs = 10
const nrhs_repeats = 10

include("../test/utils.jl")

function time_lu(input_file, mat_size, n_shared, distributed_block_rows, tile_size, logoutput=false)
    comm_world = MPI.COMM_WORLD
    comm, distributed_comm, distributed_nproc, distributed_rank, shared_comm,
        shared_nproc, shared_rank, allocate_shared_float, allocate_shared_int,
        local_win_store_float, local_win_store_int = get_comms(n_shared, true)

    timer = TimerOutput()

    if logoutput && distributed_rank == 0 && shared_rank == 0
        io = open("log-julia-timing.log", "a")
        outstream = TeeStream(io, stdout)
    else
        outstream = stdout
    end

    A = allocate_shared_float(mat_size, mat_size)
    rhs_array = allocate_shared_float(mat_size, nrhs)
    x = allocate_shared_float(mat_size)
    if distributed_rank == 0 && shared_rank == 0
        file = h5open(input_file, "r")
        mat_name = "matrix-$mat_size-$imat"
        A .= read(file[mat_name])
        vec_name = "rhs-$mat_size"
        rhs_array .= read(file[vec_name])
        close(file)
    end

    if distributed_rank == 0 && shared_rank == 0
        println(outstream, now())
        println(outstream, "benchmark: matrix=$mat_name  rhs=$vec_name  file='$input_file'")
        println(outstream, "  nproc = $(distributed_nproc * shared_nproc)  n_shared=$n_shared  distributed_block_rows = $distributed_block_rows  mat_size = $mat_size  tile_size = $tile_size  nrhs = $nrhs")
    end

    MPI.Barrier(comm_world)
    t0 = time_ns()
    A_lu = dense_lu(A, tile_size, comm, shared_comm, distributed_comm, allocate_shared_float,
                    allocate_shared_int; distributed_block_rows=distributed_block_rows,
                    skip_factorization=true, check_lu=false, timer=timer)
    MPI.Barrier(comm_world)
    t1 = time_ns()
    t_setup = (t1 - t0) / 1e9

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

        println(outstream, "Timing summary (wall-clock):")
        println(outstream, "  DenseLU setup: $t_setup s")
        println(outstream, "  LU factorisation: $t_factorisation s")
        println(outstream, "  Triangular solve:")
        println(outstream, "    min  : $t_trisolve_min s")
        println(outstream, "    mean  : $t_trisolve_mean s")
        println(outstream, "    max  : $t_trisolve_max s")
        println(outstream, "  Total (factorisation + mean solve) : $(t_factorisation + t_trisolve_mean) s")
        println(outstream)
    end

    show(outstream, timer)
    println(outstream)
    println(outstream)

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

function main()
    MPI.Init()
    nproc = MPI.Comm_size(MPI.COMM_WORLD)

    # Ensure BLAS only uses 1 thread, to avoid oversubscribing processes as we are already
    # fully parallelised.
    BLAS.set_num_threads(1)

    input_file = ARGS[1]
    mat_size = parse(Int64, ARGS[2])
    n_shared = parse(Int64, ARGS[3])
    distributed_block_rows = parse(Int64, ARGS[4])
    tile_size = parse(Int64, ARGS[5])
    if n_shared > nproc
        error("n_shared ($n_shared) > nproc ($nproc)")
    end
    if distributed_block_rows == 0
        distributed_block_rows = nothing
    end
    time_lu(input_file, mat_size, n_shared, distributed_block_rows, tile_size)
    println("repeat to remove compile timings")
    time_lu(input_file, mat_size, n_shared, distributed_block_rows, tile_size, true)

    MPI.Finalize()
    return nothing
end

main()
