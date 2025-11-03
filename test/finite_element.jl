using Primes

const ngrid = 3

function get_fe_sizes(n_element_list...)
    dim_sizes = collect((ngrid - 1) * n + 1 for n ∈ n_element_list)
    return dim_sizes, prod(dim_sizes; init=1)
end

"""
    get_distributed_slices(n_chunks, n_element_list)

Get flattened indices for dividing up an array with `n_element_list` elements in each
dimension. The first `length(n_chunks)` dimensions are divided into a number of equally
sized chunks given by the corresponding entry in `n_chunks`. The returned chunks overlap
at the element boundary point on the divide.
"""
function get_distributed_slices(n_chunks, n_element_list)
    ndims_to_divide = length(n_chunks)
    for idim ∈ 1:ndims_to_divide
        if n_element_list[idim] % n_chunks[idim] != 0
            error("Dimension $idim with $(n_element_list[idim]) elements cannot be "
                  * "divided into $(n_chunks[idim]) equally sized chunks")
        end
    end

    chunk_sizes = @views n_element_list[1:ndims_to_divide] .÷ n_chunks
    chunked_dim_sizes = n_element_list[1:ndims_to_divide]

    function get_chunked_dim_inds(i_chunk, chunk_size)
        imin = (i_chunk - 1) * chunk_size * (ngrid - 1) + 1
        imax = i_chunk * chunk_size * (ngrid - 1) + 1
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

    _, inner_dims_size = get_fe_sizes(n_element_list[ndims_to_divide+1:end]...)
    if length(n_chunks) == 1
        # It is possible to generate the slices as UnitRanges
        all_chunks = UnitRange{Int64}[]
        for i_chunk ∈ 1:n_chunks[1]
            chunk_dim_range = get_chunked_dim_inds(i_chunk, chunk_sizes[1])
            chunk_imin = chunk_dim_range.start
            chunk_imax = chunk_dim_range.stop
            imin = (chunk_imin - 1) * inner_dims_size + 1
            imax = chunk_imax * inner_dims_size
            push!(all_chunks, imin:imax)
        end
        lower_chunk_boundaries = [1:inner_dims_size]
        chunk_dim_size = chunk_sizes[1] * (ngrid - 1) + 1
        upper_chunk_boundaries = [((chunk_dim_size - 1) * inner_dims_size + 1):(chunk_dim_size * inner_dims_size)]
    else
        all_chunks = Array{Vector{Int64}}(undef, reverse(n_chunks)...)
        inner_dims_flattened_local_inds = collect(1:inner_dims_size)
        function loop_chunked_dim!(inds, i_chunks, offset, this_chunk_sizes,
                                   chunked_dim_sizes; lower_boundary_chunk_depth=nothing,
                                   upper_boundary_chunk_depth=nothing)
            # Note that because Julia uses column-major-ordered arrays, we had to reverse
            # `n_chunks` when creating `all_chunks`, so the CartesianIndex `i_chunks` is
            # 'reversed' so that the dimension that we process at this level of the
            # recursive `loop_chunked_dim!()` call is the *last* entry of `i_chunks`, but
            # the first entry of `this_chunk_sizes`.
            if length(i_chunks) == 0
                push!(inds, (offset .+ inner_dims_flattened_local_inds)...)
            else
                if lower_boundary_chunk_depth == 1
                    this_dim_inds = 1
                    lower_boundary_chunk_depth = nothing
                elseif upper_boundary_chunk_depth == 1
                    this_dim_inds = this_chunk_sizes[1] * (ngrid - 1) + 1
                    upper_boundary_chunk_depth = nothing
                else
                    this_dim_inds = get_chunked_dim_inds(i_chunks[end], this_chunk_sizes[1])
                end
                lower_boundary_chunk_depth = (lower_boundary_chunk_depth === nothing
                                              ? nothing
                                              : lower_boundary_chunk_depth - 1)
                upper_boundary_chunk_depth = (upper_boundary_chunk_depth === nothing
                                              ? nothing
                                              : upper_boundary_chunk_depth - 1)
                for i ∈ this_dim_inds
                    this_offset = offset + (i - 1) * prod(chunked_dim_sizes[2:end] .* (ngrid - 1) .+ 1; init=1) * inner_dims_size
                    loop_chunked_dim!(inds, i_chunks[1:end-1], this_offset,
                                      this_chunk_sizes[2:end], chunked_dim_sizes[2:end];
                                      lower_boundary_chunk_depth,
                                      upper_boundary_chunk_depth)
                end
            end
            return nothing
        end
        for i_chunks ∈ CartesianIndices(all_chunks)
            this_chunk = Int64[]
            loop_chunked_dim!(this_chunk, Tuple(i_chunks), 0, chunk_sizes,
                              chunked_dim_sizes)
            all_chunks[i_chunks] = this_chunk
        end
        lower_chunk_boundaries = Vector{Int64}[]
        upper_chunk_boundaries = Vector{Int64}[]
        for i ∈ 1:ndims_to_divide
            lower_inds = Int64[]
            upper_inds = Int64[]
            loop_chunked_dim!(lower_inds, Tuple(1 for _ ∈ 1:ndims_to_divide), 0,
                              chunk_sizes, chunk_sizes; lower_boundary_chunk_depth=i)
            loop_chunked_dim!(upper_inds, Tuple(1 for _ ∈ 1:ndims_to_divide), 0,
                              chunk_sizes, chunk_sizes; upper_boundary_chunk_depth=i)
            push!(lower_chunk_boundaries, lower_inds)
            push!(upper_chunk_boundaries, upper_inds)
        end
    end

    return all_chunks, lower_chunk_boundaries, upper_chunk_boundaries
end

function get_fe_like_matrix(n_element_list...; allocate_array, rng, distributed_comm,
                            shared_comm, bottom_block_ndims=1)
    # Fill top block with multi-dimensional matrix, and bottom block with 1d matrix, using
    # the first dimension of the top block.
    top_block_sizes, top_block_total = get_fe_sizes(n_element_list...)
    if bottom_block_ndims === nothing
        bottom_block_nelement_list = Int64[]
        bottom_block_sizes = Int64[]
        bottom_block_total = 0
    else
        bottom_block_nelement_list = n_element_list[1:bottom_block_ndims]
        bottom_block_sizes, bottom_block_total = get_fe_sizes(bottom_block_nelement_list...)
    end
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
        if bottom_block_ndims !== nothing
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
    end
    if shared_comm === nothing || MPI.Comm_rank(shared_comm) == 0
        MPI.Bcast!(M, distributed_comm; root=0)
    end

    return M, top_block_total, bottom_block_total
end

function finite_element_1D1V_test(n1, n2, tol; periodic=false, n_shared=1,
                                  separate_Ainv_B=false)
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
        top_chunks, top_chunk_lower_boundaries, top_chunk_upper_boundaries =
            get_distributed_slices([distributed_nproc], [n1, n2])
        top_chunk_lower_boundary = top_chunk_lower_boundaries[1]
        top_chunk_upper_boundary = top_chunk_upper_boundaries[1]
        bottom_chunks, bottom_chunk_lower_boundaries, bottom_chunk_upper_boundaries =
            get_distributed_slices([distributed_nproc], [n1])
        bottom_chunk_lower_boundary = bottom_chunk_lower_boundaries[1]
        bottom_chunk_upper_boundary = bottom_chunk_upper_boundaries[1]

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

            # Check that re-assembling our split matrices gives back the original matrix as
            # expected.
            check_M = similar(this_M)
            check_M .= 0.0
            check_A = @view check_M[1:top_block_total_size, 1:top_block_total_size]
            check_B = @view check_M[1:top_block_total_size,
                                    top_block_total_size+1:top_block_total_size+bottom_block_total_size]
            check_C = @view check_M[top_block_total_size+1:top_block_total_size+bottom_block_total_size,
                                    1:top_block_total_size]
            check_D = @view check_M[top_block_total_size+1:top_block_total_size+bottom_block_total_size,
                                    top_block_total_size+1:top_block_total_size+bottom_block_total_size]
            check_A[this_top_chunk,this_top_chunk] .= this_local_A
            check_B[this_top_chunk,this_bottom_chunk] .= this_local_B
            check_C[this_bottom_chunk,this_top_chunk] .= this_local_C
            check_D[this_bottom_chunk,this_bottom_chunk] .= this_local_D
            MPI.Allreduce!(check_M, +, distributed_comm)
            if !isapprox(M, check_M; atol=1.0e-15)
                error("M and split/reassembled M differ. Extrema: $(extrema(M .- check_M))")
            end
        end

        return this_local_A, this_local_B, this_local_C, this_local_D, this_top_chunk,
               this_bottom_chunk, top_chunks, bottom_chunks
    end

    A = @view M[1:top_block_total_size,1:top_block_total_size]
    B = @view M[1:top_block_total_size,top_block_total_size+1:end]
    C = @view M[top_block_total_size+1:end,1:top_block_total_size]
    D = @view M[top_block_total_size+1:end,top_block_total_size+1:end]
    u = @view b[1:top_block_total_size]
    v = @view b[top_block_total_size+1:end]
    x = @view z[1:top_block_total_size]
    y = @view z[top_block_total_size+1:end]

    if periodic
        n_other_dims = top_block_total_size ÷ bottom_block_total_size
        # Make RHS vectors periodic in the first dimension.
        function enforce_rhs_periodicity()
            if shared_rank == 0
                @views u[end-n_other_dims+1:end] .= u[1:n_other_dims]
                v[end] = v[1]
            end
            shared_comm !== nothing && MPI.Barrier(shared_comm)
        end
        enforce_rhs_periodicity()
    end

    # This process owns the rows/columns corresponding to top_chunk_slice and
    # bottom_chunk_slice.
    local_A, local_B, local_C, local_D, top_chunk_slice, bottom_chunk_slice,
        all_top_chunks, all_bottom_chunks = get_local_slices(M)
    local_u = @view u[top_chunk_slice]
    local_v = @view v[bottom_chunk_slice]
    local_x = @view x[top_chunk_slice]
    local_y = @view y[bottom_chunk_slice]

    if periodic
        top_chunk_global_inds = collect(top_chunk_slice)
        bottom_chunk_global_inds = collect(bottom_chunk_slice)
        if distributed_rank == distributed_nproc - 1
            @views top_chunk_global_inds[end-n_other_dims+1:end] .= 1:n_other_dims
            bottom_chunk_global_inds[end] = 1
        end
    else
        top_chunk_global_inds = top_chunk_slice
        bottom_chunk_global_inds = bottom_chunk_slice
    end
    function assemble_M(M)
        if periodic
            # Need to add up values from duplicated entries
            if shared_rank ==0 && distributed_rank == 0
                A_intermediate = A[:,1:end-n_other_dims]
                @views A_intermediate[:,1:n_other_dims] .+= A[:,end-n_other_dims+1:end]
                A_assembled = A_intermediate[1:end-n_other_dims,:]
                @views A_assembled[1:n_other_dims,:] .+= A_intermediate[end-n_other_dims+1:end,:]

                B_intermediate = B[:,1:end-1]
                @views B_intermediate[:,1] .+= B[:,end]
                B_assembled = B_intermediate[1:end-n_other_dims,:]
                @views B_assembled[1:n_other_dims,:] .+= B_intermediate[end-n_other_dims+1:end,:]

                C_intermediate = C[:,1:end-n_other_dims]
                @views C_intermediate[:,1:n_other_dims] .+= C[:,end-n_other_dims+1:end]
                C_assembled = C_intermediate[1:end-1,:]
                @views C_assembled[1,:] .+= C_intermediate[end,:]

                D_intermediate = D[:,1:end-1]
                @views D_intermediate[:,1] .+= D[:,end]
                D_assembled = D_intermediate[1:end-1,:]
                @views D_assembled[1,:] .+= D_intermediate[end,:]

                M_assembled = hvcat(2, A_assembled, B_assembled, C_assembled, D_assembled)
            else
                M_assembled = nothing
            end
        else
            M_assembled = M
        end
        return M_assembled
    end
    function assemble_b(b)
        if periodic
            # Need to add up values from duplicated entries
            if shared_rank ==0 && distributed_rank == 0
                u_assembled = @view u[1:end-n_other_dims]
                v_assembled = @view v[1:end-1]
                b_assembled = vcat(u_assembled, v_assembled)
            else
                b_assembled = nothing
            end
        else
            b_assembled = b
        end
        return b_assembled
    end

    bottom_vec_buffer = similar(y)
    global_y = similar(y)

    Alu = @views FakeMPILU(local_A, top_chunk_global_inds, top_chunk_global_inds;
                           comm=distributed_comm, shared_comm=shared_comm)

    sc = mpi_schur_complement(Alu, local_B, local_C, local_D, top_chunk_global_inds,
                              bottom_chunk_global_inds; distributed_comm=distributed_comm,
                              shared_comm=shared_comm, allocate_array=allocate_array,
                              separate_Ainv_B=separate_Ainv_B)

    function test_once(M_assembled)
        ldiv!(local_x, local_y, sc, local_u, local_v)
        shared_comm !== nothing && MPI.Barrier(shared_comm)

        b_assembled = assemble_b(b)

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
                if periodic
                    M_times_z = M * z
                    upper_check = @view M_times_z[1:top_block_total_size]
                    lower_check = @view M_times_z[top_block_total_size+1:end]
                    upper_assembled = upper_check[1:end-n_other_dims]
                    @views upper_assembled[1:n_other_dims] .+= upper_check[end-n_other_dims+1:end]
                    lower_assembled = lower_check[1:end-1]
                    lower_assembled[1] += lower_check[end]
                    M_times_z_assembled = vcat(upper_assembled, lower_assembled)

                    x_check = @view z[1:top_block_total_size]
                    y_check = @view z[top_block_total_size+1:end]
                    x_assembled = x_check[1:end-n_other_dims]
                    @test isapprox(x_assembled[1:n_other_dims],
                                   x_check[end-n_other_dims+1:end]; atol=1.0e-12)
                    y_assembled = y_check[1:end-1]
                    @test isapprox(y_assembled[1], y_check[end]; atol=1.0e-13)
                    z_assembled = vcat(x_assembled, y_assembled)
                else
                    M_times_z_assembled = M * z
                    z_assembled = z
                end
                @test isapprox(M_times_z_assembled, b_assembled; atol=tol)

                lu_sol = M_assembled \ b_assembled
                # Sanity check that tolerance is appropriate by testing solution from
                # LinearAlgebra's LU factorization.
                @test isapprox(M_assembled * lu_sol, b_assembled; atol=tol)
                # Compare our solution to the one from LinearAlgebra's LU factorization.
                @test isapprox(z_assembled, lu_sol; rtol=tol)
            end

            z .= 0.0
        end
        shared_comm !== nothing && MPI.Barrier(shared_comm)
    end

    M_assembled = assemble_M(M)

    @testset "solve" begin
        test_once(M_assembled)
    end

    @testset "change b" begin
        # Check passing a new RHS is OK
        if shared_rank == 0
            if distributed_rank == 0
                b .= rand(rng, n)
            end
            MPI.Bcast!(b, distributed_comm; root=0)
        end
        if periodic
            enforce_rhs_periodicity()
        end
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        test_once(M_assembled)
    end

    @testset "change M" begin
        # Check changing the matrix is OK
        M, _, _ = get_fe_like_matrix(n1, n2; allocate_array=allocate_array,
                                     rng=rng, distributed_comm=distributed_comm,
                                     shared_comm=shared_comm)
        A = @view M[1:top_block_total_size,1:top_block_total_size]
        B = @view M[1:top_block_total_size,top_block_total_size+1:end]
        C = @view M[top_block_total_size+1:end,1:top_block_total_size]
        D = @view M[top_block_total_size+1:end,top_block_total_size+1:end]

        M_assembled = assemble_M(M)

        if shared_rank == 0
            if distributed_rank == 0
                b .= rand(rng, n)
            end
            MPI.Bcast!(b, distributed_comm; root=0)
        end
        if periodic
            enforce_rhs_periodicity()
        end
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        local_A, local_B, local_C, local_D, _, _, _, _ = get_local_slices(M)
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        update_schur_complement!(sc, local_A, local_B, local_C, local_D)
        test_once(M_assembled)
    end

    @testset "change M, change b" begin
        # Check passing another new RHS is OK
        if shared_rank == 0
            if distributed_rank == 0
                b .= rand(rng, n)
            end
            MPI.Bcast!(b, distributed_comm; root=0)
        end
        if periodic
            enforce_rhs_periodicity()
        end
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        test_once(M_assembled)
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

function finite_element_2D1V_test(n1, n2, n3, tol; n_shared=1, periodic=false,
                                  separate_Ainv_B=false)
    distributed_comm, distributed_nproc, distributed_rank, shared_comm, shared_nproc,
        shared_rank, allocate_array, local_win_store = get_comms(n_shared)

    rng = StableRNG(2006)

    # Broadcast arrays from distributed_rank-0 so that all processes work with the same data.
    M, top_block_total_size, bottom_block_total_size =
        get_fe_like_matrix(n1, n2, n3; allocate_array=allocate_array, rng=rng,
                           distributed_comm=distributed_comm, shared_comm=shared_comm,
                           bottom_block_ndims=2)
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

    # Decide how to split up the matrix among the distributed blocks.
    distributed_factors = factor(Vector, distributed_nproc)
    n_factors = length(distributed_factors)
    first_dim_n_factors = (n_factors + 1) ÷ 2
    first_dim_distributed_nproc = prod(distributed_factors[1:first_dim_n_factors]; init=1)
    second_dim_distributed_nproc = prod(distributed_factors[first_dim_n_factors+1:end]; init=1)

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

        # Split first two dimensions among distributed ranks.
        top_chunks, top_chunk_lower_boundaries, top_chunk_upper_boundaries =
            get_distributed_slices([first_dim_distributed_nproc,
                                    second_dim_distributed_nproc],
                                   [n1, n2, n3])
        bottom_chunks, bottom_chunk_lower_boundaries, bottom_chunk_upper_boundaries =
            get_distributed_slices([first_dim_distributed_nproc,
                                    second_dim_distributed_nproc],
                                   [n1, n2])

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

            # Overlap at top-left corners.
            if distributed_rank ÷ second_dim_distributed_nproc > 0
                @views this_local_A[top_chunk_lower_boundaries[1],top_chunk_lower_boundaries[1]] .*= 0.5
                @views this_local_B[top_chunk_lower_boundaries[1],bottom_chunk_lower_boundaries[1]] .*= 0.5
                @views this_local_C[bottom_chunk_lower_boundaries[1],top_chunk_lower_boundaries[1]] .*= 0.5
                @views this_local_D[bottom_chunk_lower_boundaries[1],bottom_chunk_lower_boundaries[1]] .*= 0.5
            end
            if distributed_rank % second_dim_distributed_nproc > 0
                @views this_local_A[top_chunk_lower_boundaries[2],top_chunk_lower_boundaries[2]] .*= 0.5
                @views this_local_B[top_chunk_lower_boundaries[2],bottom_chunk_lower_boundaries[2]] .*= 0.5
                @views this_local_C[bottom_chunk_lower_boundaries[2],top_chunk_lower_boundaries[2]] .*= 0.5
                @views this_local_D[bottom_chunk_lower_boundaries[2],bottom_chunk_lower_boundaries[2]] .*= 0.5
            end
            # Overlap at bottom-right corners.
            if distributed_rank ÷ second_dim_distributed_nproc < first_dim_distributed_nproc - 1
                @views this_local_A[top_chunk_upper_boundaries[1],top_chunk_upper_boundaries[1]] .*= 0.5
                @views this_local_B[top_chunk_upper_boundaries[1],bottom_chunk_upper_boundaries[1]] .*= 0.5
                @views this_local_C[bottom_chunk_upper_boundaries[1],top_chunk_upper_boundaries[1]] .*= 0.5
                @views this_local_D[bottom_chunk_upper_boundaries[1],bottom_chunk_upper_boundaries[1]] .*= 0.5
            end
            if distributed_rank % second_dim_distributed_nproc < second_dim_distributed_nproc - 1
                @views this_local_A[top_chunk_upper_boundaries[2],top_chunk_upper_boundaries[2]] .*= 0.5
                @views this_local_B[top_chunk_upper_boundaries[2],bottom_chunk_upper_boundaries[2]] .*= 0.5
                @views this_local_C[bottom_chunk_upper_boundaries[2],top_chunk_upper_boundaries[2]] .*= 0.5
                @views this_local_D[bottom_chunk_upper_boundaries[2],bottom_chunk_upper_boundaries[2]] .*= 0.5
            end

            # Check that re-assembling our split matrices gives back the original matrix as
            # expected.
            check_M = similar(this_M)
            check_M .= 0.0
            check_A = @view check_M[1:top_block_total_size, 1:top_block_total_size]
            check_B = @view check_M[1:top_block_total_size,
                                    top_block_total_size+1:top_block_total_size+bottom_block_total_size]
            check_C = @view check_M[top_block_total_size+1:top_block_total_size+bottom_block_total_size,
                                    1:top_block_total_size]
            check_D = @view check_M[top_block_total_size+1:top_block_total_size+bottom_block_total_size,
                                    top_block_total_size+1:top_block_total_size+bottom_block_total_size]
            check_A[this_top_chunk,this_top_chunk] .= this_local_A
            check_B[this_top_chunk,this_bottom_chunk] .= this_local_B
            check_C[this_bottom_chunk,this_top_chunk] .= this_local_C
            check_D[this_bottom_chunk,this_bottom_chunk] .= this_local_D
            MPI.Allreduce!(check_M, +, distributed_comm)
            if !isapprox(M, check_M; atol=1.0e-15)
                error("M and split/reassembled M differ. Extrema: $(extrema(M .- check_M))")
            end
        end

        return this_local_A, this_local_B, this_local_C, this_local_D, this_top_chunk,
               this_bottom_chunk, top_chunks, bottom_chunks
    end

    A = @view M[1:top_block_total_size,1:top_block_total_size]
    B = @view M[1:top_block_total_size,top_block_total_size+1:end]
    C = @view M[top_block_total_size+1:end,1:top_block_total_size]
    D = @view M[top_block_total_size+1:end,top_block_total_size+1:end]
    u = @view b[1:top_block_total_size]
    v = @view b[top_block_total_size+1:end]
    x = @view z[1:top_block_total_size]
    y = @view z[top_block_total_size+1:end]

    if periodic
        top_block_sizes, _ = get_fe_sizes(n1, n2, n3)
        n_first_dim = top_block_sizes[1]
        n_second_dim = top_block_sizes[2]
        n_third_dim = top_block_sizes[3]
        # Make RHS vectors periodic in the first two dimensions.
        function enforce_rhs_periodicity()
            if shared_rank == 0
                # Periodicity in first dimension.
                @views u[end-n_third_dim*n_second_dim+1:end] .= u[1:n_third_dim*n_second_dim]
                @views v[end-n_second_dim+1:end] .= v[1:n_second_dim]
                # Periodicity in second dimension.
                for i_other ∈ 1:n_third_dim
                    @views u[n_third_dim*(n_second_dim-1)+i_other:n_third_dim*n_second_dim:end] .=
                        u[i_other:n_third_dim*n_second_dim:end]
                end
                @views v[n_second_dim:n_second_dim:end] .= v[1:n_second_dim:end]
            end
            shared_comm !== nothing && MPI.Barrier(shared_comm)
        end
        enforce_rhs_periodicity()
    end

    # This process owns the rows/columns corresponding to top_chunk_slice and
    # bottom_chunk_slice.
    local_A, local_B, local_C, local_D, top_chunk_slice, bottom_chunk_slice,
        all_top_chunks, all_bottom_chunks = get_local_slices(M)
    local_u = @view u[top_chunk_slice]
    local_v = @view v[bottom_chunk_slice]
    local_x = @view x[top_chunk_slice]
    local_y = @view y[bottom_chunk_slice]

    if periodic
        periodic_top_vector_global_inds = collect(1:top_block_total_size)
        periodic_bottom_vector_global_inds = collect(1:bottom_block_total_size)
        # Periodicity in first dimension.
        @views periodic_top_vector_global_inds[end-n_third_dim*n_second_dim+1:end] .=
            periodic_top_vector_global_inds[1:n_third_dim*n_second_dim]
        @views periodic_bottom_vector_global_inds[end-n_second_dim+1:end] .=
            periodic_bottom_vector_global_inds[1:n_second_dim]
        # Periodicity in second dimension.
        for i_other ∈ 1:n_third_dim
            @views periodic_top_vector_global_inds[n_third_dim*(n_second_dim-1)+i_other:n_third_dim*n_second_dim:end] .=
                periodic_top_vector_global_inds[i_other:n_third_dim*n_second_dim:end]
        end
        @views periodic_bottom_vector_global_inds[n_second_dim:n_second_dim:end] .=
            periodic_bottom_vector_global_inds[1:n_second_dim:end]
        # Periodicity in third dimension.
        for i_first ∈ 1:n_first_dim, i_second ∈ 1:n_second_dim
            offset = (i_first - 1) * n_second_dim * n_third_dim + (i_second - 1) * n_third_dim
            periodic_top_vector_global_inds[offset + n_third_dim] = periodic_top_vector_global_inds[offset + 1]
        end

        # The first occurence of any repeated index will always be the lower-boundary
        # point, which is the one we will want to keep in the 'assembled' matrices/vectors
        # below. The first occurence is the one kept by `unique()`, with the order of
        # entries otherwise maintained.
        unique_top_vector_global_inds = unique(periodic_top_vector_global_inds)
        unique_bottom_vector_global_inds = unique(periodic_bottom_vector_global_inds)

        top_chunk_global_inds = periodic_top_vector_global_inds[top_chunk_slice]
        bottom_chunk_global_inds = periodic_bottom_vector_global_inds[bottom_chunk_slice]

        if shared_rank == 0 && distributed_rank == 0
            top_vector_unique_inds = Int64[]
            for (i, inds) ∈ enumerate(CartesianIndices((n_third_dim, n_second_dim, n_first_dim)))
                if inds[3] == n_first_dim || inds[2] == n_second_dim
                    continue
                end
                push!(top_vector_unique_inds, i)
            end
            bottom_vector_unique_inds = Int64[]
            for (i, inds) ∈ enumerate(CartesianIndices((n_second_dim, n_first_dim)))
                if inds[2] == n_first_dim || inds[1] == n_second_dim
                    continue
                end
                push!(bottom_vector_unique_inds, i)
            end
        end
    else
        top_chunk_global_inds = top_chunk_slice
        bottom_chunk_global_inds = bottom_chunk_slice
    end
    function assemble_M(M)
        if periodic
            # Add all the upper-boundary entries (for every periodic dimension) to the
            # lower-boundary entries, then select the sub-array with the upper-boundary
            # points excluded.
            if shared_rank == 0 && distributed_rank == 0
                function add_periodic_row_entries!(x, idim, dim_sizes...)
                    for (i, inds) ∈ enumerate(CartesianIndices(dim_sizes))
                        if inds[idim] == 1
                            # Get global ind of upper boundary
                            iend = (dim_sizes[idim] - 1) * prod(dim_sizes[1:idim-1]; init=1)
                            for d ∈ 1:length(dim_sizes)
                                if d != idim
                                    iend += (inds[d] - 1) * prod(dim_sizes[1:d-1]; init=1)
                                end
                            end
                            iend += 1
                            @views x[i,:] .+= x[iend,:]
                        end
                    end
                end
                function add_periodic_column_entries!(x, idim, dim_sizes...)
                    for (i, inds) ∈ enumerate(CartesianIndices(dim_sizes))
                        if inds[idim] == 1
                            # Get global ind of upper boundary
                            iend = (dim_sizes[idim] - 1) * prod(dim_sizes[1:idim-1]; init=1)
                            for d ∈ 1:length(dim_sizes)
                                if d != idim
                                    iend += (inds[d] - 1) * prod(dim_sizes[1:d-1]; init=1)
                                end
                            end
                            iend += 1
                            @views x[:,i] .+= x[:,iend]
                        end
                    end
                end
                A_intermediate = copy(A)
                add_periodic_row_entries!(A_intermediate, 1, n_third_dim, n_second_dim, n_first_dim)
                add_periodic_row_entries!(A_intermediate, 2, n_third_dim, n_second_dim, n_first_dim)
                add_periodic_row_entries!(A_intermediate, 3, n_third_dim, n_second_dim, n_first_dim)
                A_intermediate = @view A_intermediate[unique_top_vector_global_inds,:]
                add_periodic_column_entries!(A_intermediate, 1, n_third_dim, n_second_dim, n_first_dim)
                add_periodic_column_entries!(A_intermediate, 2, n_third_dim, n_second_dim, n_first_dim)
                add_periodic_column_entries!(A_intermediate, 3, n_third_dim, n_second_dim, n_first_dim)
                A_assembled = @view A_intermediate[:,unique_top_vector_global_inds]

                B_intermediate = copy(B)
                add_periodic_row_entries!(B_intermediate, 1, n_third_dim, n_second_dim, n_first_dim)
                add_periodic_row_entries!(B_intermediate, 2, n_third_dim, n_second_dim, n_first_dim)
                add_periodic_row_entries!(B_intermediate, 3, n_third_dim, n_second_dim, n_first_dim)
                B_intermediate = @view B_intermediate[unique_top_vector_global_inds,:]
                add_periodic_column_entries!(B_intermediate, 1, n_second_dim, n_first_dim)
                add_periodic_column_entries!(B_intermediate, 2, n_second_dim, n_first_dim)
                B_assembled = @view B_intermediate[:,unique_bottom_vector_global_inds]

                C_intermediate = copy(C)
                add_periodic_row_entries!(C_intermediate, 1, n_second_dim, n_first_dim)
                add_periodic_row_entries!(C_intermediate, 2, n_second_dim, n_first_dim)
                C_intermediate = @view C_intermediate[unique_bottom_vector_global_inds,:]
                add_periodic_column_entries!(C_intermediate, 1, n_third_dim, n_second_dim, n_first_dim)
                add_periodic_column_entries!(C_intermediate, 2, n_third_dim, n_second_dim, n_first_dim)
                add_periodic_column_entries!(C_intermediate, 3, n_third_dim, n_second_dim, n_first_dim)
                C_assembled = @view C_intermediate[:,unique_top_vector_global_inds]

                D_intermediate = copy(D)
                add_periodic_row_entries!(D_intermediate, 1, n_second_dim, n_first_dim)
                add_periodic_row_entries!(D_intermediate, 2, n_second_dim, n_first_dim)
                D_intermediate = @view D_intermediate[unique_bottom_vector_global_inds,:]
                add_periodic_column_entries!(D_intermediate, 1, n_second_dim, n_first_dim)
                add_periodic_column_entries!(D_intermediate, 2, n_second_dim, n_first_dim)
                D_assembled = @view D_intermediate[:,unique_bottom_vector_global_inds]

                M_assembled = hvcat(2, A_assembled, B_assembled, C_assembled, D_assembled)
            else
                M_assembled = nothing
            end
        else
            M_assembled = M
        end
        return M_assembled
    end
    function assemble_b(b)
        if periodic
            # Need to add up values from duplicated entries
            if shared_rank ==0 && distributed_rank == 0
                u_assembled = @view u[unique_top_vector_global_inds]
                v_assembled = @view v[unique_bottom_vector_global_inds]
                b_assembled = vcat(u_assembled, v_assembled)
            else
                b_assembled = nothing
            end
        else
            b_assembled = b
        end
        return b_assembled
    end

    bottom_vec_buffer = similar(y)
    global_y = similar(y)

    Alu = @views FakeMPILU(local_A, top_chunk_global_inds, top_chunk_global_inds;
                           comm=distributed_comm, shared_comm=shared_comm)

    sc = mpi_schur_complement(Alu, local_B, local_C, local_D, top_chunk_global_inds,
                              bottom_chunk_global_inds; distributed_comm=distributed_comm,
                              shared_comm=shared_comm, allocate_array=allocate_array,
                              separate_Ainv_B=separate_Ainv_B)

    function test_once(M_assembled)
        ldiv!(local_x, local_y, sc, local_u, local_v)
        shared_comm !== nothing && MPI.Barrier(shared_comm)

        b_assembled = assemble_b(b)

        if shared_rank == 0
            buffer_x = similar(local_x)
            buffer_y = similar(local_y)
            if distributed_rank == 0
                for iproc ∈ 2:distributed_nproc
                    @views MPI.Recv!(buffer_x, distributed_comm;
                                     source=iproc-1)
                    x[all_top_chunks[iproc]] .= buffer_x
                    @views MPI.Recv!(buffer_y, distributed_comm;
                                     source=iproc-1)
                    y[all_bottom_chunks[iproc]] .= buffer_y
                end
            else
                buffer_x .= local_x
                @views MPI.Send(buffer_x, distributed_comm; dest=0)
                buffer_y .= local_y
                @views MPI.Send(buffer_y, distributed_comm; dest=0)
            end

            # Check if solution does give back original right-hand-side
            if distributed_rank == 0
                if periodic
                    function add_periodic_vector_entries!(x, idim, dim_sizes...)
                        for (i, inds) ∈ enumerate(CartesianIndices(dim_sizes))
                            if inds[idim] == 1
                                # Get global ind of upper boundary
                                iend = (dim_sizes[idim] - 1) * prod(dim_sizes[1:idim-1]; init=1)
                                for d ∈ 1:length(dim_sizes)
                                    if d != idim
                                        iend += (inds[d] - 1) * prod(dim_sizes[1:d-1]; init=1)
                                    end
                                end
                                iend += 1
                                x[i] += x[iend]
                            end
                        end
                    end
                    M_times_z = M * z
                    upper_check = @view M_times_z[1:top_block_total_size]
                    lower_check = @view M_times_z[top_block_total_size+1:end]
                    add_periodic_vector_entries!(upper_check, 1, n_third_dim, n_second_dim, n_first_dim)
                    add_periodic_vector_entries!(upper_check, 2, n_third_dim, n_second_dim, n_first_dim)
                    add_periodic_vector_entries!(upper_check, 3, n_third_dim, n_second_dim, n_first_dim)
                    upper_assembled = upper_check[unique_top_vector_global_inds]
                    add_periodic_vector_entries!(lower_check, 1, n_second_dim, n_first_dim)
                    add_periodic_vector_entries!(lower_check, 2, n_second_dim, n_first_dim)
                    lower_assembled = lower_check[unique_bottom_vector_global_inds]
                    M_times_z_assembled = vcat(upper_assembled, lower_assembled)

                    x_check = @view z[1:top_block_total_size]
                    y_check = @view z[top_block_total_size+1:end]
                    x_assembled = x_check[unique_top_vector_global_inds]
                    y_assembled = y_check[unique_bottom_vector_global_inds]
                    z_assembled = vcat(x_assembled, y_assembled)
                else
                    M_times_z_assembled = M * z
                    z_assembled = z
                end
                @test isapprox(M_times_z_assembled, b_assembled; atol=tol)

                lu_sol = M_assembled \ b_assembled
                # Sanity check that tolerance is appropriate by testing solution from
                # LinearAlgebra's LU factorization.
                @test isapprox(M_assembled * lu_sol, b_assembled; atol=tol)
                # Compare our solution to the one from LinearAlgebra's LU factorization.
                @test isapprox(z_assembled, lu_sol; rtol=tol)
            end

            z .= 0.0
        end
        shared_comm !== nothing && MPI.Barrier(shared_comm)
    end

    M_assembled = assemble_M(M)

    @testset "solve" begin
        test_once(M_assembled)
    end

    @testset "change b" begin
        # Check passing a new RHS is OK
        if shared_rank == 0
            if distributed_rank == 0
                b .= rand(rng, n)
            end
            MPI.Bcast!(b, distributed_comm; root=0)
        end
        if periodic
            enforce_rhs_periodicity()
        end
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        test_once(M_assembled)
    end

    @testset "change M" begin
        # Check changing the matrix is OK
        M, _, _ = get_fe_like_matrix(n1, n2, n3; allocate_array=allocate_array, rng=rng,
                                     distributed_comm=distributed_comm,
                                     shared_comm=shared_comm, bottom_block_ndims=2)
        A = @view M[1:top_block_total_size,1:top_block_total_size]
        B = @view M[1:top_block_total_size,top_block_total_size+1:end]
        C = @view M[top_block_total_size+1:end,1:top_block_total_size]
        D = @view M[top_block_total_size+1:end,top_block_total_size+1:end]

        M_assembled = assemble_M(M)

        if shared_rank == 0
            if distributed_rank == 0
                b .= rand(rng, n)
            end
            MPI.Bcast!(b, distributed_comm; root=0)
        end
        if periodic
            enforce_rhs_periodicity()
        end
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        local_A, local_B, local_C, local_D, _, _, _, _ = get_local_slices(M)
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        update_schur_complement!(sc, local_A, local_B, local_C, local_D)
        test_once(M_assembled)
    end

    @testset "change M, change b" begin
        # Check passing another new RHS is OK
        if shared_rank == 0
            if distributed_rank == 0
                b .= rand(rng, n)
            end
            MPI.Bcast!(b, distributed_comm; root=0)
        end
        if periodic
            enforce_rhs_periodicity()
        end
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        test_once(M_assembled)
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

# In this test there is a finite element matrix for one 'variable', without the second
# smaller coupled variable. Instead, the 'bottom vector' is chosen to be some entries that
# make the remainder of the matrix block-diagonal parts (i.e. split into disconnected
# parts).
function finite_element_3D_split_test(s1, s2, s3, tol; n_shared=1, periodic=false,
                                      separate_Ainv_B=false)
    distributed_comm, distributed_nproc, distributed_rank, shared_comm, shared_nproc,
        shared_rank, allocate_array, local_win_store = get_comms(n_shared)

    rng = StableRNG(2007)

    nelement_list = (4, 4, 4)
    ndims = length(nelement_list)

    # Broadcast arrays from distributed_rank-0 so that all processes work with the same data.
    M, n, _ = get_fe_like_matrix(nelement_list...; allocate_array=allocate_array, rng=rng,
                                 distributed_comm=distributed_comm,
                                 shared_comm=shared_comm, bottom_block_ndims=nothing)
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

    # Get the 'bottom_vector' indices corresponding to the splits s1, s2, s3 - the grid
    # points corresponding to an element boundary that, when removed, decouple the matrix
    # on either side of the split.
    function get_splitting_inds(idim, n_split)
        this_nelement = nelement_list[idim]
        if this_nelement % n_split != 0
            error("Dimension with $this_nelement elements cannot be split into $n_split "
                  * "equally sized parts.")
        end
        chunk_nelement = this_nelement ÷ n_split
        split_inds_1d = [isplit * chunk_nelement * (ngrid - 1) + 1 for isplit ∈ 1:n_split-1]
        split_inds = Int64[]
        function insert_inds!(this_dim, offset)
            if this_dim > ndims
                push!(split_inds, offset + 1)
                return nothing
            elseif this_dim == idim
                this_inds = split_inds_1d
            else
                dim_size = nelement_list[this_dim] * (ngrid - 1) + 1
                this_inds = 1:dim_size
            end
            inner_dims_size = prod(nelement_list[d] * (ngrid - 1) + 1 for d ∈ this_dim+1:ndims; init=1)
            for i ∈ this_inds
                insert_inds!(this_dim + 1, offset + (i - 1) * inner_dims_size)
            end
            return nothing
        end
        insert_inds!(1, 0)
        return split_inds
    end
    # Calculate 'splitting inds' for each dimension
    splitting_inds = Tuple(get_splitting_inds(i, s) for (i, s) ∈ enumerate((s1, s2, s3)))
    # Combine into a single list, and sort
    bottom_vector_inds = union(splitting_inds...)
    sort!(bottom_vector_inds)

    # The 'top vector' in this case is just all the indices that aren't in the 'bottom
    # vector'.
    top_vector_inds = setdiff(collect(1:n), bottom_vector_inds)

    # Decide how to split up the matrix among the distributed blocks.
    distributed_factors = factor(Vector, distributed_nproc)
    n_factors = length(distributed_factors)
    first_dim_n_factors = (n_factors + 1) ÷ 2
    first_dim_distributed_nproc = prod(distributed_factors[1:first_dim_n_factors]; init=1)
    second_dim_distributed_nproc = prod(distributed_factors[first_dim_n_factors+1:end]; init=1)

    function get_local_slices(this_M)
        # Need to extract slices, and multiply any overlaps by 0.5 so that they can be
        # recombined by adding the overlapping chunks together.

        # Split first dimension among distributed ranks.
        distributed_chunks, distributed_chunk_lower_boundaries,
        distributed_chunk_upper_boundaries =
            get_distributed_slices([first_dim_distributed_nproc,
                                    second_dim_distributed_nproc],
                                   nelement_list)
        top_chunks = [intersect(c, top_vector_inds) for c ∈ distributed_chunks]
        bottom_chunks = [intersect(c, bottom_vector_inds) for c ∈ distributed_chunks]

        this_chunk = distributed_chunks[distributed_rank+1]

        local_n = length(this_chunk)
        local_top_vector_global_inds = top_chunks[distributed_rank+1]
        local_bottom_vector_global_inds = bottom_chunks[distributed_rank+1]
        local_top_vector_local_inds = MPISchurComplements.find_local_vector_inds(local_top_vector_global_inds, this_chunk)
        local_bottom_vector_local_inds = MPISchurComplements.find_local_vector_inds(local_bottom_vector_global_inds, this_chunk)

        # Make copy here so that we do not modify the original matrix.
        this_local_M = allocate_array(local_n, local_n)
        this_local_A = @view this_local_M[local_top_vector_local_inds,local_top_vector_local_inds]
        this_local_B = @view this_local_M[local_top_vector_local_inds,local_bottom_vector_local_inds]
        this_local_C = @view this_local_M[local_bottom_vector_local_inds,local_top_vector_local_inds]
        this_local_D = @view this_local_M[local_bottom_vector_local_inds,local_bottom_vector_local_inds]

        if shared_rank == 0
            this_local_M .= @view this_M[this_chunk,this_chunk]

            # Overlap at top-left corners.
            if distributed_rank ÷ second_dim_distributed_nproc > 0
                @views this_local_M[distributed_chunk_lower_boundaries[1],distributed_chunk_lower_boundaries[1]] .*= 0.5
            end
            if distributed_rank % second_dim_distributed_nproc > 0
                @views this_local_M[distributed_chunk_lower_boundaries[2],distributed_chunk_lower_boundaries[2]] .*= 0.5
            end
            # Overlap at bottom-right corners.
            if distributed_rank ÷ second_dim_distributed_nproc < first_dim_distributed_nproc - 1
                @views this_local_M[distributed_chunk_upper_boundaries[1],distributed_chunk_upper_boundaries[1]] .*= 0.5
            end
            if distributed_rank % second_dim_distributed_nproc < second_dim_distributed_nproc - 1
                @views this_local_M[distributed_chunk_upper_boundaries[2],distributed_chunk_upper_boundaries[2]] .*= 0.5
            end

            # Check that re-assembling our split matrices gives back the original matrix as
            # expected.
            check_M = similar(this_M)
            check_M .= 0.0
            check_M[local_top_vector_global_inds,local_top_vector_global_inds] .= this_local_A
            check_M[local_top_vector_global_inds,local_bottom_vector_global_inds] .= this_local_B
            check_M[local_bottom_vector_global_inds,local_top_vector_global_inds] .= this_local_C
            check_M[local_bottom_vector_global_inds,local_bottom_vector_global_inds] .= this_local_D
            MPI.Allreduce!(check_M, +, distributed_comm)
            @test isapprox(M, check_M; atol=1.0e-15)
        end

        return this_local_A, this_local_B, this_local_C, this_local_D,
               local_top_vector_global_inds, local_bottom_vector_global_inds, top_chunks,
               bottom_chunks
    end

    u = @view b[top_vector_inds]
    v = @view b[bottom_vector_inds]
    x = @view z[top_vector_inds]
    y = @view z[bottom_vector_inds]

    if periodic
        block_sizes, _ = get_fe_sizes(nelement_list...)
        n_first_dim = block_sizes[1]
        n_second_dim = block_sizes[2]
        n_third_dim = block_sizes[3]
        # Make RHS vectors periodic in the first two dimensions.
        function enforce_rhs_periodicity()
            if shared_rank == 0
                # Periodicity in first dimension.
                @views b[end-n_third_dim*n_second_dim+1:end] .= b[1:n_third_dim*n_second_dim]
                # Periodicity in second dimension.
                for i_other ∈ 1:n_third_dim
                    @views b[n_third_dim*(n_second_dim-1)+i_other:n_third_dim*n_second_dim:end] .=
                        b[i_other:n_third_dim*n_second_dim:end]
                end
            end
            shared_comm !== nothing && MPI.Barrier(shared_comm)
        end
        enforce_rhs_periodicity()

        periodic_global_inds = collect(1:n)
        # Periodicity in first dimension.
        @views periodic_global_inds[end-n_third_dim*n_second_dim+1:end] .=
            periodic_global_inds[1:n_third_dim*n_second_dim]
        # Periodicity in second dimension.
        for i_other ∈ 1:n_third_dim
            @views periodic_global_inds[n_third_dim*(n_second_dim-1)+i_other:n_third_dim*n_second_dim:end] .=
                periodic_global_inds[i_other:n_third_dim*n_second_dim:end]
        end
        # Periodicity in third dimension.
        for i_first ∈ 1:n_first_dim, i_second ∈ 1:n_second_dim
            offset = (i_first - 1) * n_second_dim * n_third_dim + (i_second - 1) * n_third_dim
            periodic_global_inds[offset + n_third_dim] = periodic_global_inds[offset + 1]
        end

        # The first occurence of any repeated index will always be the lower-boundary
        # point, which is the one we will want to keep in the 'assembled' matrices/vectors
        # below. The first occurence is the one kept by `unique()`, with the order of
        # entries otherwise maintained.
        unique_global_inds = unique(periodic_global_inds)

        periodic_top_vector_inds = periodic_global_inds[top_vector_inds]
        periodic_bottom_vector_inds = periodic_global_inds[bottom_vector_inds]
    else
        periodic_global_inds = 1:n
        periodic_top_vector_inds = top_vector_inds
        periodic_bottom_vector_inds = bottom_vector_inds
    end

    # This process owns the rows/columns corresponding to top_chunk_slice and
    # bottom_chunk_slice.
    # Note in this test it is more convenient to slice the local chunks of u/v/x/y
    # directly out of b and z, rather than out of u/v/x/y.
    local_A, local_B, local_C, local_D, top_chunk_slice, bottom_chunk_slice,
        all_top_chunks, all_bottom_chunks = get_local_slices(M)
    local_u = @view b[top_chunk_slice]
    local_v = @view b[bottom_chunk_slice]
    local_x = @view z[top_chunk_slice]
    local_y = @view z[bottom_chunk_slice]

    function assemble_M(M)
        if periodic
            # Add all the upper-boundary entries (for every periodic dimension) to the
            # lower-boundary entries, then select the sub-array with the upper-boundary
            # points excluded.
            if shared_rank == 0 && distributed_rank == 0
                function add_periodic_row_entries!(x, idim, dim_sizes...)
                    for (i, inds) ∈ enumerate(CartesianIndices(dim_sizes))
                        if inds[idim] == 1
                            # Get global ind of upper boundary
                            iend = (dim_sizes[idim] - 1) * prod(dim_sizes[1:idim-1]; init=1)
                            for d ∈ 1:length(dim_sizes)
                                if d != idim
                                    iend += (inds[d] - 1) * prod(dim_sizes[1:d-1]; init=1)
                                end
                            end
                            iend += 1
                            @views x[i,:] .+= x[iend,:]
                        end
                    end
                end
                function add_periodic_column_entries!(x, idim, dim_sizes...)
                    for (i, inds) ∈ enumerate(CartesianIndices(dim_sizes))
                        if inds[idim] == 1
                            # Get global ind of upper boundary
                            iend = (dim_sizes[idim] - 1) * prod(dim_sizes[1:idim-1]; init=1)
                            for d ∈ 1:length(dim_sizes)
                                if d != idim
                                    iend += (inds[d] - 1) * prod(dim_sizes[1:d-1]; init=1)
                                end
                            end
                            iend += 1
                            @views x[:,i] .+= x[:,iend]
                        end
                    end
                end
                M_intermediate = copy(M)
                add_periodic_row_entries!(M_intermediate, 1, n_third_dim, n_second_dim, n_first_dim)
                add_periodic_row_entries!(M_intermediate, 2, n_third_dim, n_second_dim, n_first_dim)
                add_periodic_row_entries!(M_intermediate, 3, n_third_dim, n_second_dim, n_first_dim)
                M_intermediate = @view M_intermediate[unique_global_inds,:]
                add_periodic_column_entries!(M_intermediate, 1, n_third_dim, n_second_dim, n_first_dim)
                add_periodic_column_entries!(M_intermediate, 2, n_third_dim, n_second_dim, n_first_dim)
                add_periodic_column_entries!(M_intermediate, 3, n_third_dim, n_second_dim, n_first_dim)
                M_assembled = @view M_intermediate[:,unique_global_inds]
            else
                M_assembled = nothing
            end
        else
            M_assembled = M
        end
        return M_assembled
    end
    function assemble_b(b)
        if periodic
            # Need to add up values from duplicated entries
            if shared_rank ==0 && distributed_rank == 0
                b_assembled = @view b[unique_global_inds]
            else
                b_assembled = nothing
            end
        else
            b_assembled = b
        end
        return b_assembled
    end

    bottom_vec_buffer = similar(y)
    global_y = similar(y)

    Alu = FakeMPILU(local_A, periodic_global_inds[top_chunk_slice],
                    periodic_global_inds[top_chunk_slice]; comm=distributed_comm,
                    shared_comm=shared_comm)

    sc = mpi_schur_complement(Alu, local_B, local_C, local_D,
                              periodic_global_inds[top_chunk_slice],
                              periodic_global_inds[bottom_chunk_slice];
                              distributed_comm=distributed_comm, shared_comm=shared_comm,
                              allocate_array=allocate_array,
                              separate_Ainv_B=separate_Ainv_B)

    function test_once(M_assembled)
        ldiv!(local_x, local_y, sc, local_u, local_v)
        shared_comm !== nothing && MPI.Barrier(shared_comm)

        b_assembled = assemble_b(b)

        if shared_rank == 0
            if distributed_rank == 0
                for iproc ∈ 2:distributed_nproc
                    buffer_x = similar(local_x, size(all_top_chunks[iproc])...)
                    buffer_y = similar(local_y, size(all_bottom_chunks[iproc])...)
                    @views MPI.Recv!(buffer_x, distributed_comm;
                                     source=iproc-1)
                    z[all_top_chunks[iproc]] .= buffer_x
                    @views MPI.Recv!(buffer_y, distributed_comm;
                                     source=iproc-1)
                    z[all_bottom_chunks[iproc]] .= buffer_y
                end
            else
                buffer_x = Vector(local_x)
                @views MPI.Send(buffer_x, distributed_comm; dest=0)
                buffer_y = Vector(local_y)
                @views MPI.Send(buffer_y, distributed_comm; dest=0)
            end

            # Check if solution does give back original right-hand-side
            if distributed_rank == 0
                if periodic
                    function add_periodic_vector_entries!(x, idim, dim_sizes...)
                        for (i, inds) ∈ enumerate(CartesianIndices(dim_sizes))
                            if inds[idim] == 1
                                # Get global ind of upper boundary
                                iend = (dim_sizes[idim] - 1) * prod(dim_sizes[1:idim-1]; init=1)
                                for d ∈ 1:length(dim_sizes)
                                    if d != idim
                                        iend += (inds[d] - 1) * prod(dim_sizes[1:d-1]; init=1)
                                    end
                                end
                                iend += 1
                                x[i] += x[iend]
                            end
                        end
                    end
                    M_times_z = M * z
                    add_periodic_vector_entries!(M_times_z, 1, n_third_dim, n_second_dim, n_first_dim)
                    add_periodic_vector_entries!(M_times_z, 2, n_third_dim, n_second_dim, n_first_dim)
                    add_periodic_vector_entries!(M_times_z, 3, n_third_dim, n_second_dim, n_first_dim)
                    M_times_z_assembled = M_times_z[unique_global_inds]

                    z_assembled = z[unique_global_inds]
                else
                    M_times_z_assembled = M * z
                    z_assembled = z
                end
                @test isapprox(M_times_z_assembled, b_assembled; atol=tol)

                lu_sol = M_assembled \ b_assembled
                # Sanity check that tolerance is appropriate by testing solution from
                # LinearAlgebra's LU factorization.
                @test isapprox(M_assembled * lu_sol, b_assembled; atol=tol)
                # Compare our solution to the one from LinearAlgebra's LU factorization.
                @test isapprox(z_assembled, lu_sol; rtol=tol)
            end

            z .= 0.0
        end
        shared_comm !== nothing && MPI.Barrier(shared_comm)
    end

    M_assembled = assemble_M(M)

    @testset "solve" begin
        test_once(M_assembled)
    end

    @testset "change b" begin
        # Check passing a new RHS is OK
        if shared_rank == 0
            if distributed_rank == 0
                b .= rand(rng, n)
            end
            MPI.Bcast!(b, distributed_comm; root=0)
        end
        if periodic
            enforce_rhs_periodicity()
        end
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        test_once(M_assembled)
    end

    @testset "change M" begin
        # Check changing the matrix is OK
        M, _, _ = get_fe_like_matrix(nelement_list...; allocate_array=allocate_array,
                                     rng=rng, distributed_comm=distributed_comm,
                                     shared_comm=shared_comm, bottom_block_ndims=nothing)

        M_assembled = assemble_M(M)

        if shared_rank == 0
            if distributed_rank == 0
                b .= rand(rng, n)
            end
            MPI.Bcast!(b, distributed_comm; root=0)
        end
        if periodic
            enforce_rhs_periodicity()
        end
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        local_A, local_B, local_C, local_D, _, _, _, _ = get_local_slices(M)
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        update_schur_complement!(sc, local_A, local_B, local_C, local_D)
        test_once(M_assembled)
    end

    @testset "change M, change b" begin
        # Check passing another new RHS is OK
        if shared_rank == 0
            if distributed_rank == 0
                b .= rand(rng, n)
            end
            MPI.Bcast!(b, distributed_comm; root=0)
        end
        if periodic
            enforce_rhs_periodicity()
        end
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        test_once(M_assembled)
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
        @testset "1D1V" begin
            n_shared = 1
            while n_shared ≤ nproc
                n_distributed = nproc ÷ n_shared
                @testset "n_shared=$n_shared ($n1,$n2), tol=$tol, periodic=$periodic, separate_Ainv_B=$separate_Ainv_B" for (n1,n2,tol) ∈ (
                        (max(2, n_distributed), max(2, n_distributed), 3.0e-11),
                        (16, 8, 1.0e-9),
                        (24, 32, 3.0e-8),
                       ), periodic ∈ (false, true), separate_Ainv_B ∈ (false, true)
                    # Note that here n1 and n2 are numbers of elements, not total grid sizes.
                    # Total grid sizes are
                    # (n1*(ngrid-1)+1)*(n2*(ngrid-1)+1)=(n1*2+1)*(n2*2+1).
                    finite_element_1D1V_test(n1, n2, tol; n_shared=n_shared,
                                             periodic=periodic,
                                             separate_Ainv_B=separate_Ainv_B)
                end
                n_shared *= 2
            end
        end
        @testset "2D1V" begin
            n_shared = 1
            while n_shared ≤ nproc
                n_distributed = nproc ÷ n_shared
                @testset "n_shared=$n_shared ($n1,$n2), tol=$tol, periodic=$periodic, separate_Ainv_B=$separate_Ainv_B" for (n1,n2,tol) ∈ (
                        (max(2, n_distributed), max(2, n_distributed), 2.0e-8),
                        (8, 4, 3.0e-8),
                        (4, 12, 5.0e-9),
                       ), periodic ∈ (false, true), separate_Ainv_B ∈ (false, true)
                    finite_element_2D1V_test(n1, n2, 3, tol; n_shared=n_shared,
                                             periodic=periodic,
                                             separate_Ainv_B=separate_Ainv_B)
                end
                n_shared *= 2
            end
        end
        @testset "3D split" begin
            n_shared = 1
            while n_shared ≤ nproc
                n_distributed = nproc ÷ n_shared
                tol = 3.0e-9
                @testset "n_shared=$n_shared ($s1,$s2,$s3), periodic=$periodic, separate_Ainv_B=$separate_Ainv_B" for (s1,s2,s3) ∈ (
                        (1, 1, 2),
                        (1, 1, 4),
                        (1, 2, 1),
                        (1, 2, 2),
                        (1, 2, 4),
                        (1, 4, 1),
                        (1, 4, 2),
                        (1, 4, 4),
                        (2, 1, 1),
                        (2, 1, 2),
                        (2, 1, 4),
                        (2, 2, 1),
                        (2, 2, 2),
                        (2, 2, 4),
                        (2, 4, 1),
                        (2, 4, 2),
                        (2, 4, 4),
                        (4, 1, 1),
                        (4, 1, 2),
                        (4, 1, 4),
                        (4, 2, 1),
                        (4, 2, 2),
                        (4, 2, 4),
                        (4, 4, 1),
                        (4, 4, 2),
                        (4, 4, 4),
                       ), periodic ∈ (false, true), separate_Ainv_B ∈ (false, true)
                    finite_element_3D_split_test(s1, s2, s3, tol; n_shared=n_shared,
                                                 periodic=periodic,
                                                 separate_Ainv_B=separate_Ainv_B)
                end
                n_shared *= 2
            end
        end
    end
    return nothing
end
