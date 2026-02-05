using MPISchurComplements.DenseLUs

function dense_lu_tests()
    nproc = MPI.Comm_size(MPI.COMM_WORLD)
    # Only testing shared-memory parallelism for the DenseLU solver.
    distributed_comm, distributed_nproc, distributed_rank, shared_comm, shared_nproc,
        shared_rank, allocate_array_float, allocate_array_int, local_win_store_float,
        local_win_store_int = get_comms(nproc, true)

    rng = StableRNG(3002)

    @testset "dense_lu" begin
        @testset "m=$m, tile_size=$tile_size" for m ∈ (32, 33, 100, 128, 1009, 1024),
                                                  tile_size ∈ (2, 3, 25, 32, 90, 128)
            if tile_size > m + 5
                # If tile_size is bigger than m, the actual value of tile_size does not
                # matter, so skip what would (mostly?) be identical repeated tests.
                continue
            end

            A = allocate_array_float(m, m)
            b = allocate_array_float(m)
            x = allocate_array_float(m)

            if shared_rank == 0
                A .= rand(rng, m, m)
                b .= rand(rng, m)
            end
            MPI.Barrier(shared_comm)

            Alu = dense_lu(copy(A), tile_size, shared_comm, allocate_array_float,
                           allocate_array_int)

            function test_once()
                ldiv!(x, Alu, b)
                if shared_rank == 0
                    @test isapprox(A * x, b; norm=(x)->NaN, atol=1.0e-10)
                    @test isapprox(x, A \ b; norm=(x)->NaN, atol=2.0e-11)
                end
            end

            @testset "solve" begin
                test_once()
            end

            @testset "change b" begin
                if shared_rank == 0
                    b .= rand(rng, m)
                end
                MPI.Barrier(shared_comm)

                test_once()
            end

            @testset "change A" begin
                if shared_rank == 0
                    A .= rand(rng, m, m)
                    b .= rand(rng, m)
                end
                MPI.Barrier(shared_comm)

                lu!(Alu, copy(A))

                test_once()
            end

            @testset "change A, change b" begin
                if shared_rank == 0
                    b .= rand(rng, m)
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

    return nothing
end
