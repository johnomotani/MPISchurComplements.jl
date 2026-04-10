using MPISchurComplements.DenseLUs
using Combinatorics
using LinearAlgebra
using Primes

function dense_lu_test(n_shared)
    # Only testing shared-memory parallelism for the DenseLU solver.
    comm, distributed_comm, distributed_nproc, distributed_rank, shared_comm,
        shared_nproc, shared_rank, allocate_array_float, allocate_array_int,
        local_win_store_float, local_win_store_int = get_comms(n_shared, true)

    rng = StableRNG(3002)

    @testset "dense_lu n_shared=$n_shared" begin
        @testset "m=$m, tile_size=$tile_size" for m ∈ (32, 33, 100, 128, 295, 317, 460, 532, 604, 739, 827, 964, 1009, 1024),
                                                  tile_size ∈ (1, 2, 3, 25, 32, 90, 128)
            if tile_size > m + 5
                # If tile_size is bigger than m, the actual value of tile_size does not
                # matter, so skip what would (mostly?) be identical repeated tests.
                continue
            end
            if distributed_rank == 0 && shared_rank == 0
                println("dense_lu n_shared=$n_shared, m=$m, tile_size=$tile_size")
            end

            A = allocate_array_float(m, m)
            b = allocate_array_float(m)
            x = allocate_array_float(m)

            if shared_rank == 0 && distributed_rank == 0
                A .= rand(rng, m, m)
                # Ensure A is non-singular.
                while abs(det(A)) < 1.0e-4
                    A .= rand(rng, m, m)
                end
                b .= rand(rng, m)
            end
            if shared_rank == 0
                MPI.Bcast!(A, distributed_comm; root=0)
                MPI.Bcast!(b, distributed_comm; root=0)
            end
            MPI.Barrier(shared_comm)

            Alu = dense_lu(copy(A), tile_size, comm, shared_comm, distributed_comm,
                           allocate_array_float, allocate_array_int)

            function test_once()
                ldiv!(x, Alu, copy(b))
                # LU factorise using the row permutation calculated for Alu, to compare
                # the factors.
                check_factors_lu = lu(A[:,Alu.col_permutation], NoPivot())
                if shared_rank == 0
                    tol = 2.0e-10
                    @test isapprox(A * x, b; norm=(x)->NaN, rtol=tol, atol=tol)
                    @test isapprox(x, A \ b; norm=(x)->NaN, rtol=tol, atol=tol)
                    @test isapprox(Alu.factors, check_factors_lu.factors; norm=(x)->NaN, rtol=tol, atol=tol)
                end
            end

            @testset "solve" begin
                test_once()
            end

            @testset "change b" begin
                if shared_rank == 0 && distributed_rank == 0
                    b .= rand(rng, m)
                end
                if shared_rank == 0
                    MPI.Bcast!(b, distributed_comm; root=0)
                end
                MPI.Barrier(shared_comm)

                test_once()
            end

            @testset "change A" begin
                if shared_rank == 0 && distributed_rank == 0
                    A .= rand(rng, m, m)
                    # Ensure A is non-singular.
                    while abs(det(A)) < 1.0e-4
                        A .= rand(rng, m, m)
                    end
                    b .= rand(rng, m)
                end
                if shared_rank == 0
                    MPI.Bcast!(A, distributed_comm; root=0)
                    MPI.Bcast!(b, distributed_comm; root=0)
                end
                MPI.Barrier(shared_comm)

                lu!(Alu, copy(A))

                test_once()
            end

            @testset "change A, change b" begin
                if shared_rank == 0 && distributed_rank == 0
                    b .= rand(rng, m)
                end
                if shared_rank == 0
                    MPI.Bcast!(b, distributed_comm; root=0)
                end
                MPI.Barrier(shared_comm)

                test_once()
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

function dense_lu_tests()
    nproc = MPI.Comm_size(MPI.COMM_WORLD)
    for n_shared ∈ [prod(x) for x ∈ unique(combinations(factor(Vector, nproc)))]
        dense_lu_test(n_shared)
    end
end
