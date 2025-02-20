using LinearAlgebra
using Quadmath
using StableRNGs
using Test

using MPISchurComplements

function do_test(n1, n2, tol, float_type)
    rng = StableRNG(2001)

    function get_rand(args...)
        return float_type.(rand(rng, Float128, args...))
    end

    n = n1 + n2

    M = get_rand(n, n)
    b = get_rand(n)
    z = zeros(float_type, n)

    A = @view M[1:n1, 1:n1]
    B = @view M[1:n1, n1+1:end]
    C = @view M[n1+1:end, 1:n1]
    D = @view M[n1+1:end, n1+1:end]
    u = @view b[1:n1]
    v = @view b[n1+1:end]
    x = @view z[1:n1]
    y = @view z[n1+1:end]

    Alu = lu(A)

    sc = mpi_schur_complement(Alu, B, C, D, similar(B), similar(D), similar(x),
                              similar(x), similar(y))

    function test_once()
        ldiv!(x, y, sc, u, v)

        # Check if solution does give back original right-hand-side
        @testset "1" begin
            @test isapprox(M * z, b; atol=tol)
        end

        lu_sol = M \ b
        # Sanity check that tolerance is appropriate by testing solution from
        # LinearAlgebra's LU factorization.
        @testset "2" begin
            @test isapprox(M * lu_sol, b; atol=tol)
        end
        # Compare our solution to the one from LinearAlgebra's LU factorization.
        @testset "3" begin
            @test isapprox(z, lu_sol; rtol=tol, atol=tol)
        end
    end

    test_once()

    # Check passing a new RHS is OK
    b .= rand(rng, n)
    test_once()

    # Check changing the matrix is OK
    M .= rand(rng, n, n)
    Alu = lu(A)
    update_schur_complement!(sc, Alu, B, C, D)
    b .= rand(rng, n)
    test_once()

    # Check passing another new RHS is OK
    b .= rand(rng, n)
    test_once()
end

@testset "MPISchurComplements" begin
    @testset "$float_type ($n1,$n2), tol=$tol" for (n1,n2,tol) ∈ (
            (3, 2, 1.0e-14),
            (100, 32, 1.0e-12),
            (1000, 17, 1.0e-10),
            (1000, 129, 1.0e-10),
           ),
           float_type ∈ (
                         Float64,
                         #Float128,
                        )
        do_test(n1, n2, tol, float_type)
    end
end
