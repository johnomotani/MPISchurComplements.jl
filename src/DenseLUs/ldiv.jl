function ldiv!(A_lu::DenseLU{T}, b::AbstractVector{T}) where T
    return ldiv!(b, A_lu, b)
end

function ldiv!(x::AbstractVector{T}, A_lu::DenseLU{T}, b::AbstractVector{T}) where T
    is_root = A_lu.is_root
    row_permutation = A_lu.row_permutation
    b_permuted = A_lu.vec_buffer1
    y = A_lu.vec_buffer2
    shared_comm_rank = A_lu.shared_comm_rank
    synchronize_shared = A_lu.synchronize_shared

    # Permute the RHS, storing in buffer2. This accounts for 'row permutations' that were
    # generated/used for 'pivoting' when the L and U factors were computed.
    if is_root
        # Could parallelise this?
        @views b_permuted .= b[row_permutation]
    end

    L_solve!(y, A_lu, b_permuted)
    U_solve!(x, A_lu, y)

    # Clean up MPI requests. These should all have been completed already, so this should
    # not take any time.
    if is_root
        MPI.Waitall(A_lu.L_receive_requests)
        MPI.Waitall(A_lu.U_receive_requests)
    elseif shared_comm_rank == 0
        MPI.Waitall(A_lu.L_send_requests)
        MPI.Waitall(A_lu.U_send_requests)
        MPI.Waitall(A_lu.L_receive_requests)
        MPI.Waitall(A_lu.U_receive_requests)
    end

    return x
end

function L_solve!(y, A_lu::DenseLU{T}, b) where T
    m = A_lu.m
    n_tiles = A_lu.n_tiles
    tile_size = A_lu.tile_size
    my_L_tiles = A_lu.my_L_tiles
    my_L_tile_row_ranges = A_lu.my_L_tile_row_ranges
    my_L_tile_col_ranges = A_lu.my_L_tile_col_ranges
    diagonal_indices = A_lu.diagonal_indices
    synchronize_shared = A_lu.synchronize_shared
    L_receive_requests = A_lu.L_receive_requests
    L_send_requests = A_lu.L_send_requests
    new_column_triggers = A_lu.new_column_triggers
    step_needs_synchronize_this_block = A_lu.step_needs_synchronize_this_block
    L_rhs_update_buffer = A_lu.L_rhs_update_buffer
    shared_comm_rank = A_lu.shared_comm_rank
    distributed_comm = A_lu.distributed_comm

    if shared_comm_rank == 0
        L_rhs_update_buffer .= 0.0
    end

    if A_lu.is_root
        for step ∈ 1:length(my_L_tile_row_ranges)
            diagonal_tile = diagonal_indices[step]
            row_range = my_L_tile_row_ranges[step]
            col_range = my_L_tile_col_ranges[step]
            if diagonal_tile > 0
                # Wait to ensure that contributions from all other blocks have been added
                # to `b`.
                MPI.Wait(L_receive_requests[diagonal_tile])
                # Root process always wrote to b[tile_range] on the previous step, so no
                # need to synchronize before this calculation.
                # Still need to add this block's contributions to `b`.
                @views @. y[col_range] = b[col_range] + L_rhs_update_buffer[col_range]
                # Need the [1:length(row_range),1:length(col_range)] selection, even
                # though for most tiles this is just the full range, because the last row
                # and column may have a different size
                @views trsv!('L', 'N', 'U',
                             my_L_tiles[1:length(row_range),1:length(col_range),step],
                             y[col_range])
                if diagonal_tile < n_tiles
                    L_send_requests[diagonal_tile] = temp_Ibcast!(@view(y[col_range]),
                                                                  distributed_comm; root=0)
                    # Start MPI.Ireduce!() ready for the next diagonal tile. MPI
                    # non-blocking collective operations have to be called in the same
                    # order on all ranks
                    # (https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node126.htm),
                    # so we cannot start this operation earlier.
                    t = diagonal_tile+1
                    L_receive_requests[t] =
                        temp_Ireduce!(@view(b[(t-1)*tile_size+1:min(t*tile_size,m)]), +,
                                      distributed_comm; root=0)
                end
            else
                # Need the [1:length(row_range)] selection, even though for most tiles
                # this is just the full range, because the last row may have a different
                # size
                @views gemm!('N', 'N', -one(T), my_L_tiles[1:length(row_range),:,step],
                             y[col_range], one(T), L_rhs_update_buffer[row_range])
            end
            if step_needs_synchronize_this_block[step] == 1
                # Synchronize to avoid race conditions.
                synchronize_shared()
            end
        end
    else
        for step ∈ 1:length(my_L_tile_row_ranges)
            row_range = my_L_tile_row_ranges[step]
            col_range = my_L_tile_col_ranges[step]
            if !isempty(row_range)
                # Need the [1:length(row_range)] selection, even though for most tiles
                # this is just the full range, because the last row may have a different
                # size
                @views gemm!('N', 'N', -one(T), my_L_tiles[1:length(row_range),:,step],
                             y[col_range], one(T), L_rhs_update_buffer[row_range])
            end
            if shared_comm_rank == 0
                # `diagonal_indices[step]` is non-zero if the root process is handling a
                # diagonal tile on this step.
                maybe_diagonal_tile = diagonal_indices[step]
                if maybe_diagonal_tile > 0
                    # Data from the maybe_diagonal_tile is available, so start the
                    # MPI.Ibcast!(). Also the maybe_diagonal_tile+1 row is guaranteed to be
                    # completed, as only the root process will handle any tiles from that row
                    # from this step on, so start the MPI.Ireduce!(). MPI non-blocking
                    # collective operations have to be called in the same order on all ranks
                    # (https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node126.htm), so we
                    # have to match the order that these operations are started on the root
                    # process.
                    if maybe_diagonal_tile < n_tiles
                        L_receive_requests[maybe_diagonal_tile] =
                            temp_Ibcast!(@view(y[(maybe_diagonal_tile-1)*tile_size+1:min(maybe_diagonal_tile*tile_size, m)]),
                                         distributed_comm; root=0)
                        # We have sorted the tiles so that the shared_comm_rank=0 process
                        # always handles the lowest row in the block, so if
                        # `t` was handled on this step on this block, it was definitely
                        # handled on this rank, so we do not need to synchronize.
                        t = maybe_diagonal_tile + 1
                        L_send_requests[t] =
                            temp_Ireduce!(@view(L_rhs_update_buffer[(t-1)*tile_size+1:min(t*tile_size,m)]),
                                          +, distributed_comm; root=0)
                    end
                end
                # Ensure data required for the next tiles processed on the block has arrived.
                for tile ∈ @view new_column_triggers[:,step]
                    if tile == 0
                        # No more to do
                        break
                    end
                    MPI.Wait(L_receive_requests[tile])
                end
            end
            if step_needs_synchronize_this_block[step] == 1
                # Synchronize to avoid race conditions.
                synchronize_shared()
            end
        end
    end

    return nothing
end

function U_solve!(x, A_lu::DenseLU{T}, y) where T
    m = A_lu.m
    n_tiles = A_lu.n_tiles
    tile_size = A_lu.tile_size
    my_U_tiles = A_lu.my_U_tiles
    my_U_tile_row_ranges = A_lu.my_U_tile_row_ranges
    my_U_tile_col_ranges = A_lu.my_U_tile_col_ranges
    diagonal_indices = A_lu.diagonal_indices
    synchronize_shared = A_lu.synchronize_shared
    U_receive_requests = A_lu.U_receive_requests
    U_send_requests = A_lu.U_send_requests
    new_column_triggers = A_lu.new_column_triggers
    step_needs_synchronize_this_block = A_lu.step_needs_synchronize_this_block
    U_rhs_update_buffer = A_lu.U_rhs_update_buffer
    shared_comm_rank = A_lu.shared_comm_rank
    distributed_comm = A_lu.distributed_comm

    if shared_comm_rank == 0
        U_rhs_update_buffer .= 0.0
    end

    if A_lu.is_root
        for step ∈ 1:length(my_U_tile_row_ranges)
            diagonal_tile = diagonal_indices[step]
            row_range = my_U_tile_row_ranges[step]
            col_range = my_U_tile_col_ranges[step]
            if diagonal_tile > 0
                # Wait to ensure that contributions from all other blocks have been added
                # to `y`.
                MPI.Wait(U_receive_requests[diagonal_tile])
                # Root process always wrote to b[tile_range] on the previous step, so no
                # need to synchronize before this calculation.
                # Still need to add this block's contributions to `y`.
                @views @. x[col_range] = y[col_range] + U_rhs_update_buffer[col_range]
                # Need the [1:length(row_range),1:length(col_range)] selection, even
                # though for most tiles this is just the full range, because the last row
                # and column may have a different size
                @views trsv!('U', 'N', 'N',
                             my_U_tiles[1:length(row_range),1:length(col_range),step],
                             x[col_range])
                U_send_requests[diagonal_tile] = temp_Ibcast!(@view(x[col_range]),
                                                              distributed_comm; root=0)
                if diagonal_tile < n_tiles
                    # Start MPI.Ireduce!() ready for the next diagonal tile. MPI
                    # non-blocking collective operations have to be called in the same
                    # order on all ranks
                    # (https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node126.htm),
                    # so we cannot start this operation earlier.
                    t = diagonal_tile+1
                    U_receive_requests[t] =
                        temp_Ireduce!(@view(y[max(m-t*tile_size+1,1):m-(t-1)*tile_size]),
                                      +, distributed_comm; root=0)
                end
            else
                # Need the [1:length(row_range)] selection, even though for most tiles
                # this is just the full range, because the last row may have a different
                # size
                @views gemm!('N', 'N', -one(T), my_U_tiles[1:length(row_range),:,step],
                             x[col_range], one(T), U_rhs_update_buffer[row_range])
            end
            if step_needs_synchronize_this_block[step] == 1
                # Synchronize to avoid race conditions.
                synchronize_shared()
            end
        end
    else
        for step ∈ 1:length(my_U_tile_row_ranges)
            row_range = my_U_tile_row_ranges[step]
            col_range = my_U_tile_col_ranges[step]
            if !isempty(row_range)
                # Need the [1:length(row_range)] selection, even though for most tiles
                # this is just the full range, because the last row may have a different
                # size
                @views gemm!('N', 'N', -one(T), my_U_tiles[1:length(row_range),:,step],
                             x[col_range], one(T), U_rhs_update_buffer[row_range])
            end
            # Get data required for the next tiles processed on the block.
            if shared_comm_rank == 0
                # `diagonal_indices[step]` is non-zero if the root process is handling a
                # diagonal tile on this step.
                maybe_diagonal_tile = diagonal_indices[step]
                if maybe_diagonal_tile > 0
                    # Data from the maybe_diagonal_tile is available, so start the
                    # MPI.Ibcast!(). Also the maybe_diagonal_tile+1 row is guaranteed to be
                    # completed, as only the root process will handle any tiles from that row
                    # from this step on, so start the MPI.Ireduce!(). MPI non-blocking
                    # collective operations have to be called in the same order on all ranks
                    # (https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node126.htm), so we
                    # have to match the order that these operations are started on the root
                    # process.
                    U_receive_requests[maybe_diagonal_tile] =
                        temp_Ibcast!(@view(x[max(m-maybe_diagonal_tile*tile_size+1,1):m-(maybe_diagonal_tile-1)*tile_size]),
                                     distributed_comm; root=0)
                    if maybe_diagonal_tile < n_tiles
                        # We have sorted the tiles so that the shared_comm_rank=0 process
                        # always handles the lowest row in the block, so if
                        # `t` was handled on this step on this block, it was definitely
                        # handled on this rank, so we do not need to synchronize.
                        t = maybe_diagonal_tile + 1
                        U_send_requests[t] =
                            temp_Ireduce!(@view(U_rhs_update_buffer[max(m-t*tile_size+1,1):m-(t-1)*tile_size]),
                                          +, distributed_comm; root=0)
                    end
                end
                for tile ∈ @view new_column_triggers[:,step]
                    if tile == 0
                        # No more to do
                        break
                    end
                    MPI.Wait(U_receive_requests[tile])
                end
            end
            if step_needs_synchronize_this_block[step] == 1
                # Synchronize to avoid race conditions.
                synchronize_shared()
            end
        end
    end

    return nothing
end

# Temporarily copy functions from https://github.com/JuliaParallel/MPI.jl/pull/827, until
# that PR is merged to provide MPI.Ireduce!()
temp_Ireduce!(sendrecvbuf, op, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request(); root::Integer=Cint(0)) =
    temp_Ireduce!(sendrecvbuf, op, root, comm, req)
temp_Ireduce!(sendbuf, recvbuf, op, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request(); root::Integer=Cint(0)) =
    temp_Ireduce!(sendbuf, recvbuf, op, root, comm, req)
function temp_Ireduce!(rbuf::MPI.RBuffer, op::Union{MPI.Op,MPI.MPI_Op}, root::Integer, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request())
    # int MPI_Ireduce(const void* sendbuf, void* recvbuf, int count,
    #                 MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm,
    #                 MPI_Request* req)
    MPI.API.MPI_Ireduce(rbuf.senddata, rbuf.recvdata, rbuf.count, rbuf.datatype, op, root, comm, req)
    MPI.setbuffer!(req, rbuf)
    return req
end
temp_Ireduce!(rbuf::MPI.RBuffer, op, root::Integer, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request()) =
    temp_Ireduce!(rbuf, MPI.Op(op, eltype(rbuf)), root, comm, req)
temp_Ireduce!(sendbuf, recvbuf, op, root::Integer, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request()) =
    temp_Ireduce!(MPI.RBuffer(sendbuf, recvbuf), op, root, comm, req)
# inplace
function temp_Ireduce!(buf, op, root::Integer, comm::MPI.Comm, req::MPI.AbstractRequest=MPI.Request())
    if MPI.Comm_rank(comm) == root
        temp_Ireduce!(MPI.IN_PLACE, buf, op, root, comm, req)
    else
        temp_Ireduce!(buf, nothing, op, root, comm, req)
    end
end

# Temporarily copy functions from https://github.com/JuliaParallel/MPI.jl/pull/882, until
# that PR is merged to provide MPI.Ibcast!()
temp_Ibcast!(buf, comm::MPI.Comm; root::Integer=Cint(0)) =
    temp_Ibcast!(buf, root, comm)
function temp_Ibcast!(buf::MPI.Buffer, root::Integer, comm::MPI.Comm, req::MPI.AbstractRequest = MPI.Request())
    # int MPI_Ibcast(void *buffer, int count, MPI_Datatype datatype, int root,
    #   MPI_Comm comm, MPI_Request *request)
    MPI.API.MPI_Ibcast(buf.data, buf.count, buf.datatype, root, comm, req)
    return req
end
function temp_Ibcast!(data, root::Integer, comm::MPI.Comm)
    temp_Ibcast!(MPI.Buffer(data), root, comm)
end
