"""
Use the Schur complement technique to solve a block-2x2 matrix system, parallelised with
MPI.

Solve the matrix system
```math
\\left(\\begin{array}{cc}
A & B\\\\
C & D
\\end{array}\\right)
\\cdot\\left(\\begin{array}{c}
x\\\\
y
\\end{array}\\right)
=\\left(\\begin{array}{c}
u\\\\
v
\\end{array}\\right)
```
"""
module MPISchurComplements

export MPISchurComplement, mpi_schur_complement, update_schur_complement!, ldiv!

using LinearAlgebra
import LinearAlgebra: ldiv!
using Quadmath

mutable struct MPISchurComplement{TA,TB,TC,TSC,TSCF,TAiu,Ttv,Tbv}
    A_factorization::TA
    Ainv_dot_B::TB
    C::TC
    schur_complement::TSC
    schur_complement_factorization::TSCF
    Ainv_dot_u::TAiu
    top_vec_buffer::Ttv
    bottom_vec_buffer::Tbv
end

"""
Initialise an MPISchurComplement struct representing a 2x2 block-structured matrix
```math
\\left(\\begin{array}{cc}
A & B\\\\
C & D
\\end{array}\\right)
```

`A_factorization` should be a matrix factorization of `A`, or an object with similar
functionality, that can be passed to the `ldiv!()` function to solve a matrix system.

`B` and `D` are only used to initialize the MPISchurComplement, they are not stored. A
reference to `C` is stored.

`Ainv_dot_B` is a buffer with the same size as `B`. It does not need to be initialized.

`schur_complement` is a dense matrix with the same size as D.

`Ainv_dot_u` and `top_vec_buffer` are buffers whose length is the number of rows of `A`.
They do not need to be initialized.

`bottom_vec_buffer` is a buffer whose length is the number of rows of `D`.  It does not
need to be initialized.
"""
function mpi_schur_complement(A, B::AbstractMatrix, C::AbstractMatrix,
                              D::AbstractMatrix, Ainv_dot_B::AbstractMatrix,
                              schur_complement::AbstractMatrix,
                              Ainv_dot_u::AbstractVector, top_vec_buffer::AbstractVector,
                              bottom_vec_buffer::AbstractVector)
    @boundscheck size(A, 1) == size(B, 1) || error(BoundsError, " Rows in A_factorization do not match rows in B")
    @boundscheck size(A, 2) == size(C, 2) || error(BoundsError, " Columns in A_factorization do not match columns in C")
    @boundscheck size(D, 1) == size(C, 1) || error(BoundsError, " Rows in C do not match rows in C")
    @boundscheck size(D, 2) == size(B, 2) || error(BoundsError, " Columns in D do not match columns in B")
    @boundscheck size(D) == size(schur_complement) || error(BoundsError, " Size of D does not match size of schur_complement")
    @boundscheck size(A, 1) == size(Ainv_dot_u, 1) || error(BoundsError, " Rows in A_factorization do not match rows in Ainv_dot_u")
    @boundscheck size(A, 1) == size(top_vec_buffer, 1) || error(BoundsError, " Rows in A_factorization do not match rows in top_vec_buffer")
    @boundscheck size(D, 1) == size(bottom_vec_buffer, 1) || error(BoundsError, " Rows in D do not match rows in bottom_vec_buffer")

    A_factorization = lu(A)
    A_factorization128 = lu(Float128.(A))
    Ainv_dot_B128 = Float128.(Ainv_dot_B)
    ldiv!(Ainv_dot_B128, A_factorization128, B)
    Ainv_dot_B .= Ainv_dot_B128
    mul!(schur_complement, C, Ainv_dot_B)
    @. schur_complement = D - schur_complement
    schur_complement_factorization64 = lu(schur_complement)
    sc_factorization = MPISchurComplement(A_factorization, Ainv_dot_B, C,
                                          schur_complement,
                                          schur_complement_factorization64, Ainv_dot_u,
                                          top_vec_buffer, bottom_vec_buffer)

    return sc_factorization
end

function update_schur_complement!(sc::MPISchurComplement, A, B, C, D)
    @boundscheck size(sc.A_factorization) == size(A) || error(BoundsError, " Size of A_factorization does not match size of original A_factorization")
    @boundscheck size(sc.Ainv_dot_B) == size(B) || error(BoundsError, " Size of B does not match size of original Ainv_dot_B")
    @boundscheck size(sc.C) == size(C) || error(BoundsError, " Size of C does not match size of original C")
    @boundscheck size(sc.schur_complement) == size(D) || error(BoundsError, " Size of D does not match size of original schur_complement")

    A_factorization = lu(A)
    A_factorization128 = lu(Float128.(A))
    sc.A_factorization = A_factorization
    Ainv_dot_B128 = Float128.(sc.Ainv_dot_B)
    ldiv!(Ainv_dot_B128, A_factorization, B)
    sc.Ainv_dot_B .= Ainv_dot_B128
    sc.C = C
    mul!(sc.schur_complement, C, sc.Ainv_dot_B)
    @. sc.schur_complement = D - sc.schur_complement
    sc.schur_complement_factorization = lu!(sc.schur_complement)

    return nothing
end

function ldiv!(x, y, sc::MPISchurComplement, u, v)
    @boundscheck size(sc.top_vec_buffer) == size(u) || error(BoundsError, " Size of u does not match size of top_vec_buffer")
    @boundscheck size(sc.top_vec_buffer) == size(x) || error(BoundsError, " Size of x does not match size of top_vec_buffer")
    @boundscheck size(sc.bottom_vec_buffer) == size(v) || error(BoundsError, " Size of v does not match size of bottom_vector_buffer")
    @boundscheck size(sc.bottom_vec_buffer) == size(y) || error(BoundsError, " Size of y does not match size of bottom_vector_buffer")

    A_factorization = sc.A_factorization
    Ainv_dot_B = sc.Ainv_dot_B
    C = sc.C
    schur_complement_factorization = sc.schur_complement_factorization
    Ainv_dot_u = sc.Ainv_dot_u
    top_vec_buffer = sc.top_vec_buffer
    bottom_vec_buffer = sc.bottom_vec_buffer

    ldiv!(Ainv_dot_u, A_factorization, u)

    mul!(bottom_vec_buffer, C, Ainv_dot_u)
    @. bottom_vec_buffer = v - bottom_vec_buffer
    ldiv!(y, schur_complement_factorization, bottom_vec_buffer)

    mul!(top_vec_buffer, Ainv_dot_B, y)
    @. x = Ainv_dot_u - top_vec_buffer

    return nothing
end

end # module MPISchurComplements
