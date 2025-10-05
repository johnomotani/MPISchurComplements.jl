const ngrid = 3

function get_fe_sizes(n_element_list...)
    dim_sizes = collect((ngrid - 1) * n + 1 for n ∈ n_element_list)
    return dim_sizes, prod(dim_sizes; init=1)
end

"""
    get_distributed_slices(dim_index, n_chunks, n_element_list)

Get flattened indices for dividing up an array with `n_element_list` elements in each
dimension. The dimension given by `dim_index` is divided into `n_chunks` equally sized
pieces. The returned chunks overlap at the element boundary point on the divide.
"""
function get_distributed_slices(dim_index, n_chunks, n_element_list)
    if n_element_list[dim_index] % n_chunks != 0
        error("Dimension $dim_index with $(n_element_list[dim_index]) elements cannot be "
              * "divided into n_chunks=$n_chunks equally sized chunks")
    end

    chunk_size = n_element_list[dim_index] ÷ n_chunks

    function get_chunked_dim_inds(i_chunk, inner_dims_size)
        this_dim_imin = (i_chunk - 1) * chunk_size * (ngrid - 1) + 1
        this_dim_imax = i_chunk * chunk_size * (ngrid - 1) + 1
        imin = (this_dim_imin - 1) * inner_dims_size + 1
        imax = this_dim_imax * inner_dims_size
        return imin:imax
    end
    function get_chunk_lower_boundary_inds(inner_dims_size)
        chunk_imin = 1
        lower_boundary = chunk_imin:chunk_imin+inner_dims_size-1
        return lower_boundary
    end
    function get_chunk_upper_boundary_inds(inner_dims_size)
        chunk_dim_imax = chunk_size * (ngrid - 1) + 1
        chunk_imax = chunk_dim_imax * inner_dims_size
        upper_boundary = chunk_imax-inner_dims_size+1:chunk_imax
        return upper_boundary
    end

    if dim_index == 1
        # It is possible to generate the slices as UnitRanges
        all_chunks = UnitRange{Int64}[]
        _, other_dims_size = get_fe_sizes(n_element_list[2:end]...)
        for i_chunk ∈ 1:n_chunks
            chunk = get_chunked_dim_inds(i_chunk, other_dims_size)
            push!(all_chunks, chunk)
        end
        lower_chunk_boundary = get_chunk_lower_boundary_inds(other_dims_size)
        upper_chunk_boundary = get_chunk_upper_boundary_inds(other_dims_size)
    else
        all_chunks = Vector{Int64}[]
        outer_dims = n_element_list[1:dim_index-1]
        _, inner_dims_size = get_fe_sizes(n_element_list[dim_index+1:end]...)
        function loop_outer_dim!(inds, i_chunk, offset, this_nelements, inner_get_inds)
            if length(this_nelements) == 0
                chunk = inner_get_inds(i_chunk, inner_dims_size)
                push!(inds, offset .+ collect(chunk))
            else
                inner_elements = this_nelements[2:end]
                for i_element ∈ this_nelements[1]
                    this_offset = offset + (i_element - 1) * (ngrid - 1)
                    for igrid ∈ 1:ngrid
                        loop_outer_dim!(inds, this_offset + igrid - 1, inner_elements)
                    end
                end
            end
            return nothing
        end
        for i_chunk ∈ 1:n_chunks
            this_chunk = Int64[]
            loop_outer_dim!(this_chunk, i_chunk, 0, outer_dims, get_chunked_dim_inds)
            push!(all_chunks, this_chunk)
        end
        lower_chunk_boundary = Int64[]
        upper_chunk_boundary = Int64[]
        loop_outer_dim!(lower_chunk_boundary, 1, 0, outer_dims,
                        get_chunk_lower_boundary_inds)
        loop_outer_dim!(upper_chunk_boundary, 1, 0, outer_dims,
                        get_chunk_upper_boundary_inds)
    end

    return all_chunks, lower_chunk_boundary, upper_chunk_boundary
end

function get_fe_like_matrix(n_element_list...; allocate_array, rng, distributed_comm,
                            shared_comm, bottom_block_ndims=1)
    # Fill top block with multi-dimensional matrix, and bottom block with 1d matrix, using
    # the first dimension of the top block.
    top_block_sizes, top_block_total = get_fe_sizes(n_element_list...)
    bottom_block_nelement_list = n_element_list[1:bottom_block_ndims]
    bottom_block_sizes, bottom_block_total = get_fe_sizes(bottom_block_nelement_list...)
    total_size = top_block_total + bottom_block_total
    M = allocate_array(total_size, total_size)

    function fill_fe_recursive!(M_subblock, n_element_row_list, n_element_column_list)
        if length(n_element_row_list) == 0 && length(n_element_column_list) == 0
            M_subblock[1,1] = rand(rng)
            return nothing
        elseif length(n_element_row_list) == 0
            # Keep going, but just fill the same point in the row.
            row_slice = ones(Int64, n_element_column_list[1])
            column_slice = 1:n_element_column_list[1]
            sub_block_row_size = 1
            sub_block_column_size = prod(n_element_column_list[2:end] .* (ngrid - 1) .+ 1;
                                         init=1)
            ngrid_row = 1
            ngrid_column = ngrid
        elseif length(n_element_column_list) == 0
            # Keep going, but just fill the same point in the column.
            column_slice = ones(Int64, n_element_row_list[1])
            row_slice = 1:n_element_row_list[1]
            sub_block_row_size = prod(n_element_row_list[2:end] .* (ngrid - 1) .+ 1;
                                      init=1)
            sub_block_column_size = 1
            ngrid_row = ngrid
            ngrid_column = 1
        else
            if n_element_row_list[1] != n_element_column_list[1]
                error("all matching elements of rows and columns should have the same "
                      * "size")
            end
            row_slice = 1:n_element_row_list[1]
            column_slice = 1:n_element_column_list[1]
            sub_block_row_size = prod(n_element_row_list[2:end] .* (ngrid - 1) .+ 1;
                                      init=1)
            sub_block_column_size = prod(n_element_column_list[2:end] .* (ngrid - 1) .+ 1;
                                         init=1)
            ngrid_row = ngrid
            ngrid_column = ngrid
        end
        other_elements_row_list = n_element_row_list[2:end]
        other_elements_column_list = n_element_column_list[2:end]
        for (i_row, i_column) ∈ zip(row_slice, column_slice)
            imin_row = (i_row - 1) * (ngrid_row - 1) + 1
            imax_row = imin_row + ngrid_row - 1
            imin_column = (i_column - 1) * (ngrid_column - 1) + 1
            imax_column = imin_column + ngrid_column - 1
            # Fill every point within this element
            for j ∈ imin_column:imax_column, i ∈ imin_row:imax_row
                this_row_slice = (i-1)*sub_block_row_size+1:i*sub_block_row_size
                this_column_slice = (j-1)*sub_block_column_size+1:j*sub_block_column_size
                this_sub_block = @view M_subblock[this_row_slice,this_column_slice]
                fill_fe_recursive!(this_sub_block, other_elements_row_list,
                                   other_elements_column_list)
            end
        end
    end

    if ((distributed_comm == MPI.COMM_NULL || MPI.Comm_rank(distributed_comm)) == 0
            && (shared_comm === nothing || MPI.Comm_rank(shared_comm) == 0))
        top_block_slice = 1:top_block_total
        bottom_block_slice = top_block_total+1:total_size
        # Fill 'A' block
        @views fill_fe_recursive!(M[top_block_slice,top_block_slice], n_element_list,
                                  n_element_list)
        # Fill 'B' block
        @views fill_fe_recursive!(M[top_block_slice,bottom_block_slice], n_element_list,
                                  bottom_block_nelement_list)
        # Fill 'C' block
        @views fill_fe_recursive!(M[bottom_block_slice,top_block_slice],
                                  bottom_block_nelement_list, n_element_list)
        # Fill 'D' block
        @views fill_fe_recursive!(M[bottom_block_slice,bottom_block_slice],
                                  bottom_block_nelement_list, bottom_block_nelement_list)
    end
    if shared_comm === nothing || MPI.Comm_rank(shared_comm) == 0
        MPI.Bcast!(M, distributed_comm; root=0)
    end

    return M, top_block_total, bottom_block_total
end

function finite_element_1D1V_test(n1, n2, tol; n_shared=1, separate_Ainv_B=false)
    distributed_comm, distributed_nproc, distributed_rank, shared_comm, shared_nproc,
        shared_rank, allocate_array, local_win_store = get_comms(n_shared)

    rng = StableRNG(2004)

    # Broadcast arrays from distributed_rank-0 so that all processes work with the same data.
    M, top_block_total_size, bottom_block_total_size =
        get_fe_like_matrix(n1, n2; allocate_array=allocate_array, rng=rng,
                           distributed_comm=distributed_comm, shared_comm=shared_comm)
    n = top_block_total_size + bottom_block_total_size
    b = allocate_array(n)
    z = allocate_array(n)
    if distributed_rank == 0 && shared_rank == 0
        b .= rand(rng, n)
        z .= 0.0
    end
    if shared_rank == 0
        MPI.Bcast!(b, distributed_comm; root=0)
        MPI.Bcast!(z, distributed_comm; root=0)
    end
    shared_comm !== nothing && MPI.Barrier(shared_comm)

    function get_local_slices(this_M)
        # Need to extract slices, and multiply any overlaps by 0.5 so that they can be
        # recombined by adding the overlapping chunks together.

        this_A = @view this_M[1:top_block_total_size, 1:top_block_total_size]
        this_B = @view this_M[1:top_block_total_size,
                              top_block_total_size+1:top_block_total_size+bottom_block_total_size]
        this_C = @view this_M[top_block_total_size+1:top_block_total_size+bottom_block_total_size,
                              1:top_block_total_size]
        this_D = @view this_M[top_block_total_size+1:top_block_total_size+bottom_block_total_size,
                              top_block_total_size+1:top_block_total_size+bottom_block_total_size]

        # Split first dimension among distributed ranks.
        top_chunks, top_chunk_lower_boundary, top_chunk_upper_boundary =
            get_distributed_slices(1, distributed_nproc, [n1, n2])
        bottom_chunks, bottom_chunk_lower_boundary, bottom_chunk_upper_boundary =
            get_distributed_slices(1, distributed_nproc, [n1])

        this_top_chunk = top_chunks[distributed_rank+1]
        this_bottom_chunk = bottom_chunks[distributed_rank+1]

        # Make copies here so that we do not modify the original matrix.
        local_ntop = length(this_top_chunk)
        local_nbottom = length(this_bottom_chunk)
        this_local_A = allocate_array(local_ntop,local_ntop)
        this_local_B = allocate_array(local_ntop,local_nbottom)
        this_local_C = allocate_array(local_nbottom,local_ntop)
        this_local_D = allocate_array(local_nbottom,local_nbottom)

        if shared_rank == 0
            this_local_A .= this_A[this_top_chunk,this_top_chunk]
            this_local_B .= this_B[this_top_chunk,this_bottom_chunk]
            this_local_C .= this_C[this_bottom_chunk,this_top_chunk]
            this_local_D .= this_D[this_bottom_chunk,this_bottom_chunk]

            if distributed_rank != 0
                # Overlap at top-left corners.
                this_local_A[top_chunk_lower_boundary,top_chunk_lower_boundary] *= 0.5
                this_local_B[top_chunk_lower_boundary,bottom_chunk_lower_boundary] *= 0.5
                this_local_C[bottom_chunk_lower_boundary,top_chunk_lower_boundary] *= 0.5
                this_local_D[bottom_chunk_lower_boundary,bottom_chunk_lower_boundary] *= 0.5
            end
            #if distributed_rank != distributed_nproc - 1
            if distributed_rank != MPI.Comm_size(distributed_comm) - 1
                # Overlap at bottom-right corners.
                this_local_A[top_chunk_upper_boundary,top_chunk_upper_boundary] *= 0.5
                this_local_B[top_chunk_upper_boundary,bottom_chunk_upper_boundary] *= 0.5
                this_local_C[bottom_chunk_upper_boundary,top_chunk_upper_boundary] *= 0.5
                this_local_D[bottom_chunk_upper_boundary,bottom_chunk_upper_boundary] *= 0.5
            end
        end

        return this_local_A, this_local_B, this_local_C, this_local_D, this_top_chunk,
               this_bottom_chunk, top_chunks, bottom_chunks
    end

    u = @view b[1:top_block_total_size]
    v = @view b[top_block_total_size+1:end]
    x = @view z[1:top_block_total_size]
    y = @view z[top_block_total_size+1:end]

    # This process owns the rows/columns corresponding to top_chunk_slice and
    # bottom_chunk_slice.
    local_A, local_B, local_C, local_D, top_chunk_slice, bottom_chunk_slice,
        all_top_chunks, all_bottom_chunks = get_local_slices(M)
    local_u = @view u[top_chunk_slice]
    local_v = @view v[bottom_chunk_slice]
    local_x = @view x[top_chunk_slice]
    local_y = @view y[bottom_chunk_slice]

    bottom_vec_buffer = similar(y)
    global_y = similar(y)

    Alu = @views FakeMPILU(local_A, top_chunk_slice, top_chunk_slice;
                           comm=distributed_comm, shared_comm=shared_comm)

    sc = mpi_schur_complement(Alu, local_B, local_C, local_D, top_chunk_slice,
                              bottom_chunk_slice; distributed_comm=distributed_comm,
                              shared_comm=shared_comm, allocate_array=allocate_array,
                              separate_Ainv_B=separate_Ainv_B)

    function test_once()
        ldiv!(local_x, local_y, sc, local_u, local_v)
        shared_comm !== nothing && MPI.Barrier(shared_comm)

        if shared_rank == 0
            if distributed_rank == 0
                for iproc ∈ 2:distributed_nproc
                    @views MPI.Recv!(x[all_top_chunks[iproc]], distributed_comm;
                                     source=iproc-1)
                    @views MPI.Recv!(y[all_bottom_chunks[iproc]], distributed_comm;
                                     source=iproc-1)
                end
            else
                @views MPI.Send(local_x, distributed_comm; dest=0)
                @views MPI.Send(local_y, distributed_comm; dest=0)
            end

            # Check if solution does give back original right-hand-side
            if distributed_rank == 0
                @test isapprox(M * z, b; atol=tol)

                lu_sol = M \ b
                # Sanity check that tolerance is appropriate by testing solution from
                # LinearAlgebra's LU factorization.
                @test isapprox(M * lu_sol, b; atol=tol)
                # Compare our solution to the one from LinearAlgebra's LU factorization.
                @test isapprox(z, lu_sol; rtol=tol)
            end

            z .= 0.0
        end
        shared_comm !== nothing && MPI.Barrier(shared_comm)
    end

    @testset "solve" begin
        test_once()
    end

    @testset "change b" begin
        # Check passing a new RHS is OK
        if shared_rank == 0
            if distributed_rank == 0
                b .= rand(rng, n)
            end
            MPI.Bcast!(b, distributed_comm; root=0)
        end
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        test_once()
    end

    @testset "change M" begin
        # Check changing the matrix is OK
        M, _, _ = get_fe_like_matrix(n1, n2; allocate_array=allocate_array,
                                     rng=rng, distributed_comm=distributed_comm,
                                     shared_comm=shared_comm)
        if shared_rank == 0
            if distributed_rank == 0
                b .= rand(rng, n)
            end
            MPI.Bcast!(b, distributed_comm; root=0)
        end
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        local_A, local_B, local_C, local_D, _, _, _, _ = get_local_slices(M)
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        update_schur_complement!(sc, local_A, local_B, local_C, local_D)
        test_once()
    end

    @testset "change M, change b" begin
        # Check passing another new RHS is OK
        if shared_rank == 0
            if distributed_rank == 0
                b .= rand(rng, n)
            end
            MPI.Bcast!(b, distributed_comm; root=0)
        end
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        test_once()
    end
    if local_win_store !== nothing
        # Free the MPI.Win objects, because if they are free'd by the garbage collector
        # it may cause an MPI error or hang.
        for w ∈ local_win_store
            MPI.free(w)
        end
        resize!(local_win_store, 0)
    end
end

function finite_element_tests()
    @testset "finite element" begin
        nproc = MPI.Comm_size(MPI.COMM_WORLD)
        n_shared = 1
        while n_shared ≤ nproc
            n_distributed = nproc ÷ n_shared
            @testset "n_shared=$n_shared ($n1,$n2), tol=$tol, separate_Ainv_B=$separate_Ainv_B" for (n1,n2,tol) ∈ (
                    (max(2, n_distributed), max(2, n_distributed), 1.0e-11),
                    (16, 8, 1.0e-9),
                    (24, 32, 3.0e-8),
                   ), separate_Ainv_B ∈ (true, false)
                # Note that here n1 and n2 are numbers of elements, not total grid sizes.
                # Total grid sizes are
                # (n1*(ngrid-1)+1)*(n2*(ngrid-1)+1)=(n1*2+1)*(n2*2+1).
                @testset "finite element 1D1V" begin
                    finite_element_1D1V_test(n1, n2, tol; n_shared=n_shared,
                                             separate_Ainv_B=separate_Ainv_B)
                end
            end
            n_shared *= 2
        end
    end
    return nothing
end
