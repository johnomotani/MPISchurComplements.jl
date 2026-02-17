function dense_matrix_test(n1, n2, tol; n_shared=1, with_comm=false, use_sparse=true,
                           separate_Ainv_B=false, use_unitrange=true, parallel_schur=true)
    comm, distributed_comm, distributed_nproc, distributed_rank, shared_comm,
        shared_nproc, shared_rank, allocate_array_float, allocate_array_int,
        local_win_store_float, local_win_store_int = get_comms(n_shared, with_comm)

    if use_unitrange
        convert_range = identity
    else
        convert_range = collect
    end

    n = n1 + n2

    local_n1, rem = divrem(n1, distributed_nproc)
    rem != 0 && error("distributed_nproc=$distributed_nproc does not divide n1=$n1")

    local_top_vec_range = (distributed_rank * local_n1 + 1):(distributed_rank + 1)*local_n1

    local_n2, rem = divrem(n2, distributed_nproc)
    rem != 0 && error("distributed_nproc=$distributed_nproc does not divide n2=$n2")

    local_bottom_vec_range = (distributed_rank * local_n2 + 1):(distributed_rank + 1)*local_n2

    # Broadcast arrays from distributed_rank-0 so that all processes work with the same data.
    M = allocate_array_float(n, n)
    b = allocate_array_float(n)
    z = allocate_array_float(n)
    if distributed_rank == 0 && shared_rank == 0
        rng = StableRNG(2001)

        M .= rand(rng, n, n)
        b .= rand(rng, n)
        z .= 0.0
    end
    if shared_rank == 0
        MPI.Bcast!(M, distributed_comm; root=0)
        MPI.Bcast!(b, distributed_comm; root=0)
        MPI.Bcast!(z, distributed_comm; root=0)
    end

    A = @view M[1:n1, 1:n1]
    B = @view M[1:n1, n1+1:end]
    C = @view M[n1+1:end, 1:n1]
    D = @view M[n1+1:end, n1+1:end]
    u = @view b[1:n1]
    v = @view b[n1+1:end]
    x = @view z[1:n1]
    y = @view z[n1+1:end]

    # This process owns the *columns* corresponding to local_top_vec_range and
    # local_bottom_vec_range. The rows are distributed between different processes.
    local_A = @view A[local_top_vec_range,:]
    local_B = @view B[local_top_vec_range,:]
    local_C = @view C[:,local_top_vec_range]
    local_D = @view D[local_bottom_vec_range,:]
    local_u = @view u[local_top_vec_range]
    local_v = @view v[local_bottom_vec_range]
    local_x = @view x[local_top_vec_range]
    local_y = @view y[local_bottom_vec_range]

    Alu = FakeMPILU(local_A, convert_range(local_top_vec_range); comm=distributed_comm,
                    shared_comm=shared_comm)

    owned_top_vector_entries = distributed_rank*local_n1+1:(distributed_rank+1)*local_n1
    owned_bottom_vector_entries = distributed_rank*local_n2+1:(distributed_rank+1)*local_n2

    bottom_vec_buffer = similar(y)
    global_y = similar(y)

    sc = mpi_schur_complement(Alu, copy(local_B), copy(local_C), copy(local_D),
                              convert_range(owned_top_vector_entries),
                              convert_range(owned_bottom_vector_entries);
                              B_global_column_range=convert_range(1:n2),
                              C_global_row_range=convert_range(1:n2),
                              D_global_column_range=convert_range(1:n2),
                              comm=comm, shared_comm=shared_comm,
                              distributed_comm=distributed_comm,
                              allocate_shared_float=allocate_array_float,
                              allocate_shared_int=allocate_array_int,
                              use_sparse=use_sparse, separate_Ainv_B=separate_Ainv_B,
                              parallel_schur=parallel_schur)

    function test_once()
        ldiv!(local_x, local_y, sc, local_u, local_v)
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        if shared_rank == 0
            MPI.Barrier(distributed_comm)
            MPI.Reduce!(z, +, distributed_comm)

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
        if shared_rank == 0
            if distributed_rank == 0
                M .= rand(rng, n, n)
                b .= rand(rng, n)
            end
            MPI.Bcast!(M, distributed_comm; root=0)
            MPI.Bcast!(b, distributed_comm; root=0)
        end
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        update_schur_complement!(sc, copy(local_A), copy(local_B), copy(local_C),
                                 copy(local_D))
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
end

function sparse_matrix_test(n1, n2, tol; n_shared=1, with_comm=false, use_sparse=true,
                            separate_Ainv_B=false, use_unitrange=true,
                            parallel_schur=true)
    comm, distributed_comm, distributed_nproc, distributed_rank, shared_comm,
        shared_nproc, shared_rank, allocate_array_float, allocate_array_int,
        local_win_store_float, local_win_store_int = get_comms(n_shared, with_comm)

    if use_unitrange
        convert_range = identity
    else
        convert_range = collect
    end

    n = n1 + n2

    local_n1, rem = divrem(n1, distributed_nproc)
    rem != 0 && error("distributed_nproc=$distributed_nproc does not divide n1=$n1")

    local_top_vec_range = (distributed_rank * local_n1 + 1):(distributed_rank + 1)*local_n1

    local_n2, rem = divrem(n2, distributed_nproc)
    rem != 0 && error("distributed_nproc=$distributed_nproc does not divide n2=$n2")

    local_bottom_vec_range = (distributed_rank * local_n2 + 1):(distributed_rank + 1)*local_n2

    # Boundaries in indices along 'n1' axes of B and C where the sparsity pattern changes.
    sparsity_boundaries = [1, max(n1 ÷ 4, 1), max(n1 ÷ 2, 1), max((3 * n1) ÷ 4, 1), n1+1]
    # Minimum/maximum non-zero index on the 'n2' axes of B and C for each region in
    # sparsity_boundaries.
    top_imin = [1, max(n1 ÷ 8, 1), max((3*n1) ÷ 8, 1), max((5*n1) ÷ 8, 1)]
    top_imax = [max((3*n1) ÷ 8, 1), max((5*n1) ÷ 8, 1), max((7*n1) ÷ 8, 1), n1]
    bottom_imin = [1, max(n2 ÷ 8, 1), max((3*n2) ÷ 8, 1), max((5*n2) ÷ 8, 1), n2+1]
    bottom_imax = [max((3*n2) ÷ 8, 1), max((5*n2) ÷ 8, 1), max((7*n2) ÷ 8, 1), n2]

    function sparsify_M!(this_M)
        this_A = @view this_M[1:n1, 1:n1]
        this_B = @view this_M[1:n1, n1+1:n1+n2]
        this_C = @view this_M[n1+1:n1+n2, 1:n1]
        this_D = @view this_M[n1+1:n1+n2, n1+1:n1+n2]

        for i ∈ 1:4
            this_A[sparsity_boundaries[i]:sparsity_boundaries[i+1]-1,1:top_imin[i]-1] .= 0.0
            this_A[sparsity_boundaries[i]:sparsity_boundaries[i+1]-1,top_imax[i]+1:end] .= 0.0

            this_B[sparsity_boundaries[i]:sparsity_boundaries[i+1]-1,1:bottom_imin[i]-1] .= 0.0
            this_B[sparsity_boundaries[i]:sparsity_boundaries[i+1]-1,bottom_imax[i]+1:end] .= 0.0

            this_C[1:bottom_imin[i]-1,sparsity_boundaries[i]:sparsity_boundaries[i+1]-1] .= 0.0
            this_C[bottom_imax[i]+1:end,sparsity_boundaries[i]:sparsity_boundaries[i+1]-1] .= 0.0

            # Slightly abuse `bottom_imin`, using to play the role for D that sparsity_boundaries
            # plays for B.
            this_D[bottom_imin[i]:bottom_imin[i+1]-1,1:bottom_imin[i]-1] .= 0.0
            this_D[bottom_imin[i]:bottom_imin[i+1]-1,bottom_imax[i]+1:end] .= 0.0
        end
    end

    # Broadcast arrays from distributed_rank-0 so that all processes work with the same data.
    M = allocate_array_float(n, n)
    b = allocate_array_float(n)
    z = allocate_array_float(n)
    if distributed_rank == 0 && shared_rank == 0
        rng = StableRNG(2002)

        M .= rand(rng, n, n)
        sparsify_M!(M)
        b .= rand(rng, n)
        z .= 0.0
    end
    if shared_rank == 0
        MPI.Bcast!(M, distributed_comm; root=0)
        MPI.Bcast!(b, distributed_comm; root=0)
        MPI.Bcast!(z, distributed_comm; root=0)
    end

    A = @view M[1:n1, 1:n1]
    B = @view M[1:n1, n1+1:end]
    C = @view M[n1+1:end, 1:n1]
    D = @view M[n1+1:end, n1+1:end]
    u = @view b[1:n1]
    v = @view b[n1+1:end]
    x = @view z[1:n1]
    y = @view z[n1+1:end]

    # This process owns the *columns* corresponding to local_top_vec_range and
    # local_bottom_vec_range. The rows are distributed between different processes.
    local_A = @view A[local_top_vec_range,:]
    local_B = @view B[local_top_vec_range,:]
    local_C = @view C[:,local_top_vec_range]
    local_D = @view D[local_bottom_vec_range,:]
    local_u = @view u[local_top_vec_range]
    local_v = @view v[local_bottom_vec_range]
    local_x = @view x[local_top_vec_range]
    local_y = @view y[local_bottom_vec_range]

    owned_top_vector_entries = distributed_rank*local_n1+1:(distributed_rank+1)*local_n1
    owned_bottom_vector_entries = distributed_rank*local_n2+1:(distributed_rank+1)*local_n2

    # Find local non-zero index ranges for B, C, D.
    local_top_imin = -1
    local_top_imax = -1
    local_bottom_imin = -1
    local_bottom_imax = -1
    D_local_imin = -1
    D_local_imax = -1
    for i ∈ 1:4
        if owned_top_vector_entries.start ≥ sparsity_boundaries[i] && owned_top_vector_entries.start < sparsity_boundaries[i+1]
            local_top_imin = top_imin[i]
            local_bottom_imin = bottom_imin[i]
        end
        if owned_top_vector_entries.stop ≥ sparsity_boundaries[i] && owned_top_vector_entries.stop < sparsity_boundaries[i+1]
            local_top_imax = top_imax[i]
            local_bottom_imax = bottom_imax[i]
        end
        if owned_bottom_vector_entries.start ≥ bottom_imin[i] && owned_bottom_vector_entries.start < bottom_imin[i+1]
            D_local_imin = bottom_imin[i]
        end
        if owned_bottom_vector_entries.stop ≥ bottom_imin[i] && owned_bottom_vector_entries.stop < bottom_imin[i+1]
            D_local_imax = bottom_imax[i]
        end
    end
    local_top_irange = local_top_imin:local_top_imax
    local_bottom_irange = local_bottom_imin:local_bottom_imax
    D_local_irange = D_local_imin:D_local_imax

    bottom_vec_buffer = similar(y)
    global_y = similar(y)

    Alu = @views FakeMPILU(local_A[:,local_top_irange],
                           convert_range(local_top_vec_range),
                           convert_range(local_top_irange); comm=distributed_comm,
                           shared_comm=shared_comm)

    sc = mpi_schur_complement(Alu, local_B[:,local_bottom_irange],
                              local_C[local_bottom_irange,:], local_D[:,D_local_irange],
                              convert_range(owned_top_vector_entries),
                              convert_range(owned_bottom_vector_entries);
                              B_global_column_range=convert_range(local_bottom_irange),
                              C_global_row_range=convert_range(local_bottom_irange),
                              D_global_column_range=convert_range(D_local_irange),
                              comm=comm, shared_comm=shared_comm,
                              distributed_comm=distributed_comm,
                              allocate_shared_float=allocate_array_float,
                              allocate_shared_int=allocate_array_int,
                              use_sparse=use_sparse, separate_Ainv_B=separate_Ainv_B,
                              parallel_schur=parallel_schur)

    function test_once()
        ldiv!(local_x, local_y, sc, local_u, local_v)
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        if shared_rank == 0
            MPI.Barrier(distributed_comm)
            MPI.Reduce!(z, +, distributed_comm)

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
        if shared_rank == 0
            if distributed_rank == 0
                M .= rand(rng, n, n)
                sparsify_M!(M)
                b .= rand(rng, n)
            end
            MPI.Bcast!(M, distributed_comm; root=0)
            MPI.Bcast!(b, distributed_comm; root=0)
        end
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        update_schur_complement!(sc, local_A[:,local_top_irange],
                                 local_B[:,local_bottom_irange],
                                 local_C[local_bottom_irange,:],
                                 local_D[:,D_local_irange])
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
end

function overlap_matrix_test(local_n1, local_n2, tol; n_shared=1, with_comm=false,
                             use_sparse=true, separate_Ainv_B=false, use_unitrange=true,
                             add_index_gap=false, parallel_schur=true)
    comm, distributed_comm, distributed_nproc, distributed_rank, shared_comm,
        shared_nproc, shared_rank, allocate_array_float, allocate_array_int,
        local_win_store_float, local_win_store_int = get_comms(n_shared, with_comm)

    if use_unitrange
        convert_range = identity
    else
        convert_range = collect
    end

    rng = StableRNG(2003)

    n1 = (local_n1 - 1) * distributed_nproc + 1
    n2 = (local_n2 - 1) * distributed_nproc + 1
    n = n1 + n2

    # Set up overlapping 'owned' ranges
    local_top_vec_range = (distributed_rank * (local_n1 - 1) + 1):((distributed_rank + 1) * (local_n1 - 1) + 1)
    local_bottom_vec_range = (distributed_rank * (local_n2 - 1) + 1):((distributed_rank + 1) * (local_n2 - 1) + 1)

    if add_index_gap
        local_top_vec_range = collect(local_top_vec_range)
        local_bottom_vec_range = collect(local_bottom_vec_range)
        nominal_local_top_vec_range = [i > 2*n1÷3 ? i+20 : i for i ∈ local_top_vec_range]
        nominal_local_bottom_vec_range = [i > 2*n2÷3 ? i+10 : i for i ∈ local_bottom_vec_range]
    else
        nominal_local_top_vec_range = local_top_vec_range
        nominal_local_bottom_vec_range = local_bottom_vec_range
    end

    function initialize_M!(this_M)
        this_A = @view this_M[1:n1, 1:n1]
        this_B = @view this_M[1:n1, n1+1:n1+n2]
        this_C = @view this_M[n1+1:n1+n2, 1:n1]
        this_D = @view this_M[n1+1:n1+n2, n1+1:n1+n2]

        this_M .= 0.0

        for i ∈ 1:distributed_nproc
            this_top_vec_range = ((i - 1) * (local_n1 - 1) + 1):(i * (local_n1 - 1) + 1)
            this_bottom_vec_range = ((i - 1) * (local_n2 - 1) + 1):(i * (local_n2 - 1) + 1)
            ntop = length(this_top_vec_range)
            nbottom = length(this_bottom_vec_range)

            this_A[this_top_vec_range,this_top_vec_range] .= rand(rng, ntop, ntop)
            this_B[this_top_vec_range,this_bottom_vec_range] .= rand(rng, ntop, nbottom)
            this_C[this_bottom_vec_range,this_top_vec_range] .= rand(rng, nbottom, ntop)
            this_D[this_bottom_vec_range,this_bottom_vec_range] .= rand(rng, nbottom, nbottom)
        end
    end
    function get_local_slices(this_M)
        # Need to extract slices, and multiply any overlaps by 0.5 so that they can be
        # recombined by adding the overlapping chunks together.

        this_A = @view this_M[1:n1, 1:n1]
        this_B = @view this_M[1:n1, n1+1:n1+n2]
        this_C = @view this_M[n1+1:n1+n2, 1:n1]
        this_D = @view this_M[n1+1:n1+n2, n1+1:n1+n2]

        # Make copies here so that we do not modify the original matrix.
        local_ntop = length(local_top_vec_range)
        local_nbottom = length(local_bottom_vec_range)
        this_local_A = allocate_array_float(local_ntop,local_ntop)
        this_local_B = allocate_array_float(local_ntop,local_nbottom)
        this_local_C = allocate_array_float(local_nbottom,local_ntop)
        this_local_D = allocate_array_float(local_nbottom,local_nbottom)

        if shared_rank == 0
            this_local_A .= this_A[local_top_vec_range,local_top_vec_range]
            this_local_B .= this_B[local_top_vec_range,local_bottom_vec_range]
            this_local_C .= this_C[local_bottom_vec_range,local_top_vec_range]
            this_local_D .= this_D[local_bottom_vec_range,local_bottom_vec_range]

            if distributed_rank != 0
                # Overlap at top-left corners.
                this_local_A[1,1] *= 0.5
                this_local_B[1,1] *= 0.5
                this_local_C[1,1] *= 0.5
                this_local_D[1,1] *= 0.5
            end
            #if distributed_rank != distributed_nproc - 1
            if distributed_rank != MPI.Comm_size(distributed_comm) - 1
                # Overlap at bottom-right corners.
                this_local_A[end,end] *= 0.5
                this_local_B[end,end] *= 0.5
                this_local_C[end,end] *= 0.5
                this_local_D[end,end] *= 0.5
            end
        end

        return this_local_A, this_local_B, this_local_C, this_local_D
    end

    # Broadcast arrays from distributed_rank-0 so that all processes work with the same data.
    M = allocate_array_float(n, n)
    b = allocate_array_float(n)
    z = allocate_array_float(n)
    if distributed_rank == 0 && shared_rank == 0
        initialize_M!(M)
        b .= rand(rng, n)
        z .= 0.0
    end
    if shared_rank == 0
        MPI.Bcast!(M, distributed_comm; root=0)
        MPI.Bcast!(b, distributed_comm; root=0)
        MPI.Bcast!(z, distributed_comm; root=0)
    end
    shared_comm !== nothing && MPI.Barrier(shared_comm)

    A = @view M[1:n1, 1:n1]
    B = @view M[1:n1, n1+1:end]
    C = @view M[n1+1:end, 1:n1]
    D = @view M[n1+1:end, n1+1:end]
    u = @view b[1:n1]
    v = @view b[n1+1:end]
    x = @view z[1:n1]
    y = @view z[n1+1:end]

    # This process owns the *columns* corresponding to local_top_vec_range and
    # local_bottom_vec_range. The rows are distributed between different processes.
    local_A, local_B, local_C, local_D = get_local_slices(M)
    local_u = @view u[local_top_vec_range]
    local_v = @view v[local_bottom_vec_range]
    local_x = @view x[local_top_vec_range]
    local_y = @view y[local_bottom_vec_range]

    bottom_vec_buffer = similar(y)
    global_y = similar(y)

    Alu = @views FakeMPILU(local_A, convert_range(nominal_local_top_vec_range),
                           convert_range(nominal_local_top_vec_range); comm=distributed_comm,
                           shared_comm=shared_comm)

    sc = mpi_schur_complement(Alu, local_B, local_C, local_D,
                              convert_range(nominal_local_top_vec_range),
                              convert_range(nominal_local_bottom_vec_range);
                              comm=comm, shared_comm=shared_comm,
                              distributed_comm=distributed_comm,
                              allocate_shared_float=allocate_array_float,
                              allocate_shared_int=allocate_array_int,
                              use_sparse=use_sparse, separate_Ainv_B=separate_Ainv_B,
                              parallel_schur=parallel_schur)

    function test_once()
        ldiv!(local_x, local_y, sc, local_u, local_v)
        shared_comm !== nothing && MPI.Barrier(shared_comm)

        if shared_rank == 0
            # Drop overlapping parts of solution.
            if distributed_rank != 0
                local_x[1] = 0.0
                local_y[1] = 0.0
            end

            MPI.Barrier(distributed_comm)
            MPI.Reduce!(z, +, distributed_comm)

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
        if shared_rank == 0
            if distributed_rank == 0
                initialize_M!(M)
                b .= rand(rng, n)
            end
            MPI.Bcast!(M, distributed_comm; root=0)
            MPI.Bcast!(b, distributed_comm; root=0)
        end
        shared_comm !== nothing && MPI.Barrier(shared_comm)
        local_A, local_B, local_C, local_D = get_local_slices(M)
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
end

function simple_matrix_tests()
    @testset "simple matrix" begin
        nproc = MPI.Comm_size(MPI.COMM_WORLD)
        if nproc == 1
            # Test prime vector sizes - easier to do in serial.
            @testset "($n1,$n2), tol=$tol with_comm=$with_comm, use_sparse=$use_sparse, separate_Ainv_B=$separate_Ainv_B, use_unitrange=$use_unitrange, parallel_schur=$parallel_schur" for (n1,n2,tol) ∈ (
                    (3, 2, 5.0e-14),
                    (100, 32, 2.0e-10),
                    (1000, 17, 1.0e-8),
                    (1000, 129, 1.0e-8),
                   ), with_comm ∈ (false, true), use_sparse ∈ (true, false), separate_Ainv_B ∈ (true, false), use_unitrange ∈ (true, false), parallel_schur ∈ (true, false)
                if !use_sparse && separate_Ainv_B
                    continue
                end
                println("simple_matrix ($n1,$n2), tol=$tol with_comm=$with_comm, use_sparse=$use_sparse, separate_Ainv_B=$separate_Ainv_B, use_unitrange=$use_unitrange, parallel_schur=$parallel_schur")
                @testset "dense" begin
                    dense_matrix_test(n1, n2, tol; with_comm=with_comm,
                                      use_sparse=use_sparse,
                                      separate_Ainv_B=separate_Ainv_B,
                                      use_unitrange=use_unitrange,
                                      parallel_schur=parallel_schur)
                end
                @testset "use_sparse" begin
                    sparse_matrix_test(n1, n2, tol; with_comm=with_comm,
                                       use_sparse=use_sparse,
                                       separate_Ainv_B=separate_Ainv_B,
                                       use_unitrange=use_unitrange,
                                       parallel_schur=parallel_schur)
                end
                @testset "overlap" begin
                    # As there is only one process here, there is no 'overlap', but run
                    # the test anyway as a sanity-check.
                    overlap_matrix_test(n1 + 1, n2 + 1, tol; use_sparse=use_sparse,
                                        separate_Ainv_B=separate_Ainv_B,
                                        use_unitrange=use_unitrange,
                                        parallel_schur=parallel_schur)
                end
                @testset "overlap with index gap" begin
                    # As there is only one process here, there is no 'overlap', but run
                    # the test anyway as a sanity-check.
                    overlap_matrix_test(n1 + 1, n2 + 1, tol; use_sparse=use_sparse,
                                        separate_Ainv_B=separate_Ainv_B,
                                        use_unitrange=use_unitrange,
                                        parallel_schur=parallel_schur, add_index_gap=true)
                end
            end
        elseif nproc % 2 != 0
            error("Distributed MPI test only implemented for distributed_nproc=2^n, distributed_nproc<32")
        else
            n_shared = 1
            while n_shared ≤ nproc
                n_distributed = nproc ÷ n_shared
                @testset "n_shared=$n_shared ($n1,$n2), tol=$tol, use_sparse=$use_sparse, separate_Ainv_B=$separate_Ainv_B, use_unitrange=$use_unitrange, parallel_schur=$parallel_schur" for (n1,n2,tol) ∈ (
                        (128, 32, 4.0e-10),
                        (1024, 32, 1.0e-8),
                        (1024, 128, 3.0e-7),
                       ), use_sparse ∈ (true, false), separate_Ainv_B ∈ (true, false), use_unitrange ∈ (true, false), parallel_schur ∈ (true, false)
                    if !use_sparse && separate_Ainv_B
                        continue
                    end
                    println("simple_matrix n_shared=$n_shared ($n1,$n2), tol=$tol, use_sparse=$use_sparse, separate_Ainv_B=$separate_Ainv_B, use_unitrange=$use_unitrange, parallel_schur=$parallel_schur")
                    @testset "dense" begin
                        dense_matrix_test(n1, n2, tol; n_shared=n_shared,
                                          use_sparse=use_sparse,
                                          separate_Ainv_B=separate_Ainv_B,
                                          use_unitrange=use_unitrange,
                                          parallel_schur=parallel_schur)
                    end
                    @testset "use_sparse" begin
                        sparse_matrix_test(n1, n2, tol; n_shared=n_shared,
                                           use_sparse=use_sparse,
                                           separate_Ainv_B=separate_Ainv_B,
                                           use_unitrange=use_unitrange,
                                           parallel_schur=parallel_schur)
                    end
                    @testset "overlap" begin
                        overlap_matrix_test(n1 ÷ n_distributed + 1,
                                            n2 ÷ n_distributed + 1, tol;
                                            n_shared=n_shared, use_sparse=use_sparse,
                                            separate_Ainv_B=separate_Ainv_B,
                                            use_unitrange=use_unitrange,
                                            parallel_schur=parallel_schur)
                    end
                    @testset "overlap with index gap" begin
                        overlap_matrix_test(n1 ÷ n_distributed + 1,
                                            n2 ÷ n_distributed + 1, tol;
                                            n_shared=n_shared, use_sparse=use_sparse,
                                            separate_Ainv_B=separate_Ainv_B,
                                            use_unitrange=use_unitrange,
                                            add_index_gap=true,
                                            parallel_schur=parallel_schur)
                    end
                end
                n_shared *= 2
            end
        end
    end
    return nothing
end
