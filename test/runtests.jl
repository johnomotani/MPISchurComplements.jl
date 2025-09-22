using LinearAlgebra
using MPI
using StableRNGs
using Test

using MPISchurComplements

include("FakeMPILUs.jl")

function dense_matrix_test(n1, n2, tol)
    distributed_comm = MPI.COMM_WORLD
    distributed_nproc = MPI.Comm_size(distributed_comm)
    distributed_rank = MPI.Comm_rank(distributed_comm)

    n = n1 + n2

    local_n1, rem = divrem(n1, distributed_nproc)
    rem != 0 && error("distributed_nproc=$distributed_nproc does not divide n1=$n1")

    local_top_vec_range = (distributed_rank * local_n1 + 1):(distributed_rank + 1)*local_n1

    local_n2, rem = divrem(n2, distributed_nproc)
    rem != 0 && error("distributed_nproc=$distributed_nproc does not divide n2=$n2")

    local_bottom_vec_range = (distributed_rank * local_n2 + 1):(distributed_rank + 1)*local_n2

    local_n = local_n1 + local_n2

    # Broadcast arrays from distributed_rank-0 so that all processes work with the same data.
    if distributed_rank == 0
        rng = StableRNG(2001)

        M = rand(rng, n, n)
        b = rand(rng, n)
        z = zeros(n)
        MPI.Bcast!(M, distributed_comm; root=0)
        MPI.Bcast!(b, distributed_comm; root=0)
        MPI.Bcast!(z, distributed_comm; root=0)
    else
        M = zeros(n, n)
        b = zeros(n)
        z = zeros(n)
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
    local_B = @view B[local_top_vec_range,:]
    local_C = @view C[:,local_top_vec_range]
    local_D = @view D[local_bottom_vec_range,:]
    local_u = @view u[local_top_vec_range]
    local_v = @view v[local_bottom_vec_range]
    local_x = @view x[local_top_vec_range]
    local_y = @view y[local_bottom_vec_range]

    if distributed_rank == 0
        Alu = FakeMPILU(A, length(local_x); comm=distributed_comm)
    else
        Alu = FakeMPILU(nothing, length(local_x); comm=distributed_comm)
    end

    owned_top_vector_entries = distributed_rank*local_n1+1:(distributed_rank+1)*local_n1
    owned_bottom_vector_entries = distributed_rank*local_n2+1:(distributed_rank+1)*local_n2

    bottom_vec_buffer = similar(y)
    global_y = similar(y)

    sc = mpi_schur_complement(Alu, copy(local_B), 1:n2, copy(local_C), 1:n2,
                              copy(local_D), 1:n2, owned_top_vector_entries,
                              owned_bottom_vector_entries, distributed_comm)

    function test_once()
        ldiv!(local_x, local_y, sc, local_u, local_v)
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

    test_once()

    # Check passing a new RHS is OK
    if distributed_rank == 0
        b .= rand(rng, n)
    end
    MPI.Bcast!(b, distributed_comm; root=0)
    test_once()

    # Check changing the matrix is OK
    if distributed_rank == 0
        M .= rand(rng, n, n)
    end
    MPI.Bcast!(M, distributed_comm; root=0)
    update_schur_complement!(sc, copy(A), copy(local_B), copy(local_C), copy(local_D))
    if distributed_rank == 0
        b .= rand(rng, n)
    end
    MPI.Bcast!(b, distributed_comm; root=0)
    test_once()

    # Check passing another new RHS is OK
    if distributed_rank == 0
        b .= rand(rng, n)
    end
    MPI.Bcast!(b, distributed_comm; root=0)
    test_once()
end

function sparse_matrix_test(n1, n2, tol)
    distributed_comm = MPI.COMM_WORLD
    distributed_nproc = MPI.Comm_size(distributed_comm)
    distributed_rank = MPI.Comm_rank(distributed_comm)

    n = n1 + n2

    local_n1, rem = divrem(n1, distributed_nproc)
    rem != 0 && error("distributed_nproc=$distributed_nproc does not divide n1=$n1")

    local_top_vec_range = (distributed_rank * local_n1 + 1):(distributed_rank + 1)*local_n1

    local_n2, rem = divrem(n2, distributed_nproc)
    rem != 0 && error("distributed_nproc=$distributed_nproc does not divide n2=$n2")

    local_bottom_vec_range = (distributed_rank * local_n2 + 1):(distributed_rank + 1)*local_n2

    local_n = local_n1 + local_n2

    # Boundaries in indices along 'n1' axes of B and C where the sparsity pattern changes.
    sparsity_boundaries = [1, max(n1 ÷ 4, 1), max(n1 ÷ 2, 1), max((3 * n1) ÷ 4, 1), n1+1]
    # Minimum/maximum non-zero index on the 'n2' axes of B and C for each region in
    # sparsity_boundaries.
    imin = [1, max(n2 ÷ 8, 1), max((3*n2) ÷ 8, 1), max((5*n2) ÷ 8, 1), n2+1]
    imax = [max((3*n2) ÷ 8, 1), max((5*n2) ÷ 8, 1), max((7*n2) ÷ 8, 1), n2]

    function sparsify_M!(this_M)
        this_B = @view this_M[1:n1, n1+1:n1+n2]
        this_C = @view this_M[n1+1:n1+n2, 1:n1]
        this_D = @view this_M[n1+1:n1+n2, n1+1:n1+n2]

        for i ∈ 1:4
            this_B[sparsity_boundaries[i]:sparsity_boundaries[i+1]-1,1:imin[i]-1] .= 0.0
            this_B[sparsity_boundaries[i]:sparsity_boundaries[i+1]-1,imax[i]+1:end] .= 0.0

            this_C[1:imin[i]-1,sparsity_boundaries[i]:sparsity_boundaries[i+1]-1] .= 0.0
            this_C[imax[i]+1:end,sparsity_boundaries[i]:sparsity_boundaries[i+1]-1] .= 0.0

            # Slightly abuse `imin`, using to play the role for D that sparsity_boundaries
            # plays for B.
            this_D[imin[i]:imin[i+1]-1,1:imin[i]-1] .= 0.0
            this_D[imin[i]:imin[i+1]-1,imax[i]+1:end] .= 0.0
        end
    end

    # Broadcast arrays from distributed_rank-0 so that all processes work with the same data.
    if distributed_rank == 0
        rng = StableRNG(2001)

        M = rand(rng, n, n)
        sparsify_M!(M)
        b = rand(rng, n)
        z = zeros(n)
        MPI.Bcast!(M, distributed_comm; root=0)
        MPI.Bcast!(b, distributed_comm; root=0)
        MPI.Bcast!(z, distributed_comm; root=0)
    else
        M = zeros(n, n)
        b = zeros(n)
        z = zeros(n)
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
    local_B = @view B[local_top_vec_range,:]
    local_C = @view C[:,local_top_vec_range]
    local_D = @view D[local_bottom_vec_range,:]
    local_u = @view u[local_top_vec_range]
    local_v = @view v[local_bottom_vec_range]
    local_x = @view x[local_top_vec_range]
    local_y = @view y[local_bottom_vec_range]

    if distributed_rank == 0
        Alu = FakeMPILU(A, length(local_x); comm=distributed_comm)
    else
        Alu = FakeMPILU(nothing, length(local_x); comm=distributed_comm)
    end

    owned_top_vector_entries = distributed_rank*local_n1+1:(distributed_rank+1)*local_n1
    owned_bottom_vector_entries = distributed_rank*local_n2+1:(distributed_rank+1)*local_n2

    # Find local non-zero index ranges for B, C, D.
    local_imin = -1
    local_imax = -1
    D_local_imin = -1
    D_local_imax = -1
    for i ∈ 1:4
        if owned_top_vector_entries.start ≥ sparsity_boundaries[i] && owned_top_vector_entries.start < sparsity_boundaries[i+1]
            local_imin = imin[i]
        end
        if owned_top_vector_entries.stop ≥ sparsity_boundaries[i] && owned_top_vector_entries.stop < sparsity_boundaries[i+1]
            local_imax = imax[i]
        end
        if owned_bottom_vector_entries.start ≥ imin[i] && owned_bottom_vector_entries.start < imin[i+1]
            D_local_imin = imin[i]
        end
        if owned_bottom_vector_entries.stop ≥ imin[i] && owned_bottom_vector_entries.stop < imin[i+1]
            D_local_imax = imax[i]
        end
    end
    local_irange = local_imin:local_imax
    D_local_irange = D_local_imin:D_local_imax

    bottom_vec_buffer = similar(y)
    global_y = similar(y)

    sc = mpi_schur_complement(Alu, local_B[:,local_irange], local_irange,
                              local_C[local_irange,:], local_irange,
                              local_D[:,D_local_irange], D_local_irange,
                              owned_top_vector_entries, owned_bottom_vector_entries,
                              distributed_comm)

    function test_once()
        ldiv!(local_x, local_y, sc, local_u, local_v)
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

    test_once()

    # Check passing a new RHS is OK
    if distributed_rank == 0
        b .= rand(rng, n)
    end
    MPI.Bcast!(b, distributed_comm; root=0)
    test_once()

    # Check changing the matrix is OK
    if distributed_rank == 0
        M .= rand(rng, n, n)
        sparsify_M!(M)
    end
    MPI.Bcast!(M, distributed_comm; root=0)
    update_schur_complement!(sc, copy(A), local_B[:,local_irange],
                             local_C[local_irange,:], local_D[:,D_local_irange])
    if distributed_rank == 0
        b .= rand(rng, n)
    end
    MPI.Bcast!(b, distributed_comm; root=0)
    test_once()

    # Check passing another new RHS is OK
    if distributed_rank == 0
        b .= rand(rng, n)
    end
    MPI.Bcast!(b, distributed_comm; root=0)
    test_once()
end

function runtests()
    if !MPI.Initialized()
        MPI.Init()
    end
    @testset "MPISchurComplements" begin
        if MPI.Comm_size(MPI.COMM_WORLD) == 1
            # Test prime vector sizes - easier to do in serial.
            @testset "($n1,$n2), tol=$tol" for (n1,n2,tol) ∈ (
                    (3, 2, 1.0e-14),
                    (100, 32, 1.0e-10),
                    (1000, 17, 1.0e-8),
                    (1000, 129, 1.0e-8),
                   )
                @testset "dense" begin
                    dense_matrix_test(n1, n2, tol)
                end
                @testset "sparse" begin
                    sparse_matrix_test(n1, n2, tol)
                end
            end
        elseif MPI.Comm_size(MPI.COMM_WORLD) % 2 != 0
            error("Distributed MPI test only implemented for distributed_nproc=2^n, distributed_nproc<32")
        else
            # Test prime vector sizes - easier to do in serial.
            @testset "($n1,$n2), tol=$tol" for (n1,n2,tol) ∈ (
                    (128, 32, 4.0e-10),
                    (1024, 32, 1.0e-8),
                    (1024, 128, 3.0e-7),
                   )
                @testset "dense" begin
                    dense_matrix_test(n1, n2, tol)
                end
                @testset "sparse" begin
                    sparse_matrix_test(n1, n2, tol)
                end
            end
        end
    end
end
runtests()
