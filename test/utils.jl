function get_comms(shared_nproc, with_comm=false)
    if shared_nproc == 1 && !with_comm
        distributed_comm = MPI.COMM_WORLD
        distributed_nproc = MPI.Comm_size(distributed_comm)
        distributed_rank = MPI.Comm_rank(distributed_comm)
        shared_comm = nothing
        shared_rank = 0
    else
        nproc = MPI.Comm_size(MPI.COMM_WORLD)
        rank = MPI.Comm_rank(MPI.COMM_WORLD)
        distributed_nproc, rem = divrem(nproc, shared_nproc)
        if rem != 0
            error("shared_nproc=$shared_nproc does not divide nproc=$nproc")
        end
        distributed_rank, shared_rank = divrem(rank, shared_nproc)
        shared_comm = MPI.Comm_split(MPI.COMM_WORLD, distributed_rank, shared_rank)
        if shared_rank == 0
            distributed_color = 0
        else
            distributed_color = nothing
        end
        distributed_comm = MPI.Comm_split(MPI.COMM_WORLD, distributed_color,
                                          distributed_rank)
    end

    local_win_store = nothing
    if shared_comm === nothing && !with_comm
        allocate_array = (args...)->zeros(Float64, args...)
    else
        local_win_store = MPI.Win[]
        allocate_array = (dims...)->begin
            if shared_rank == 0
                dims_local = dims
            else
                dims_local = Tuple(0 for _ ∈ dims)
            end
            win, array_temp = MPI.Win_allocate_shared(Array{Float64}, dims_local,
                                                      shared_comm)
            array = MPI.Win_shared_query(Array{Float64}, dims, win; rank=0)
            push!(local_win_store, win)
            return array
        end
    end

    return distributed_comm, distributed_nproc, distributed_rank, shared_comm,
           shared_nproc, shared_rank, allocate_array, local_win_store
end
