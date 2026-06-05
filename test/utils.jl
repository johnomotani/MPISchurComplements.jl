function get_comms(shared_nproc, with_comm=false)
    comm = MPI.COMM_WORLD
    nproc = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)
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

    local_win_store_float = nothing
    if shared_comm == MPI.COMM_SELF && !with_comm
        allocate_array_float = (args...)->zeros(Float64, args...)
    else
        local_win_store_float = MPI.Win[]
        allocate_array_float = (dims...)->begin
            if shared_rank == 0
                dims_local = dims
            else
                dims_local = Tuple(0 for _ ∈ dims)
            end
            win, array_temp = MPI.Win_allocate_shared(Array{Float64}, dims_local,
                                                      shared_comm)
            array = MPI.Win_shared_query(Array{Float64}, dims, win; rank=0)
            push!(local_win_store_float, win)
            if shared_rank == 0
                array .= NaN
            end
            MPI.Barrier(shared_comm)
            return array
        end
    end

    local_win_store_int = nothing
    if shared_comm == MPI.COMM_SELF && !with_comm
        allocate_array_int = (args...)->zeros(Float64, args...)
    else
        local_win_store_int = MPI.Win[]
        allocate_array_int = (dims...)->begin
            if shared_rank == 0
                dims_local = dims
            else
                dims_local = Tuple(0 for _ ∈ dims)
            end
            win, array_temp = MPI.Win_allocate_shared(Array{Int64}, dims_local,
                                                      shared_comm)
            array = MPI.Win_shared_query(Array{Int64}, dims, win; rank=0)
            push!(local_win_store_int, win)
            if shared_rank == 0
                array .= typemin(Int64)
            end
            MPI.Barrier(shared_comm)
            return array
        end
    end

    return comm, distributed_comm, distributed_nproc, distributed_rank, shared_comm,
           shared_nproc, shared_rank, allocate_array_float, allocate_array_int,
           local_win_store_float, local_win_store_int
end

# This is a fairly stupid thing to do, but having no structural zeros in the sparse matrix
# buffers ensures that they all have a compatible set of entries.
function get_full_sparse_matrix_copy(M, allocate_shared_float=nothing, shared_rank=-1)
    m, n = size(M)
    rowinds = vcat((1:m for _ ∈ 1:n)...)
    colinds = vcat(([j for _ ∈ 1:m] for j ∈ 1:n)...)
    if allocate_shared_float === nothing
        return sparse(rowinds, colinds, copy(vec(M)), m, n)
    else
        temp = sparse(rowinds, colinds, zeros(length(rowinds)), m, n)
        nzvals = allocate_shared_float(length(rowinds))
        if shared_rank == 0
            nzvals .= vec(M)
        end
        return FixedSparseCSC(m, n, temp.colptr, temp.rowval, nzvals)
    end
end

function get_full_shared_sparse_matrix(m, n, allocate_shared_float)
    rowinds = vcat((1:m for _ ∈ 1:n)...)
    colinds = vcat(([j for _ ∈ 1:m] for j ∈ 1:n)...)
    temp = sparse(rowinds, colinds, zeros(length(rowinds)), m, n)
    return FixedSparseCSC(m, n, temp.colptr, temp.rowval, allocate_shared_float(length(rowinds)))
end
