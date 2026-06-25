module MPISchurComplements
@doc read(joinpath(dirname(@__DIR__), "README.md"), String) MPISchurComplements

export MPISchurComplement, mpi_schur_complement, update_schur_complement!, ldiv!

using LinearAlgebra
import LinearAlgebra: ldiv!
using MPI
using MPIDenseLUs
using SparseArrays
using SparseArrays: FixedSparseCSC, AbstractSparseMatrixCSC
using SparseMatricesCSR
using TimerOutputs

const Trange = Vector{Int64}

macro sc_timeit(timer, name, expr)
    return quote
        if $(esc(timer)) === nothing
            $(esc(expr))
        else
            @timeit $(esc(timer)) $(esc(name)) $(esc(expr))
        end
    end
end

"""
    MPISchurComplementAFactorization{T} <: Factorization{T}

External packages can implement `Factorization` solvers of this type, so that when they
are used for the \$A^{-1}\\cdot B\$ operation, they are called with the special
`ldiv_Bmatrix!()` function instead of the generic `ldiv!()`. This allows the
externally-implemented solver to specialise on any known structure of the `B` block of the
matrix.
"""
abstract type MPISchurComplementAFactorization{T} <: Factorization{T} end

"""
    ldiv_Bmatrix!(Alu::MPISchurComplementAFactorization, u)

Special version of `ldiv!()` called by `MPISchurComplements` for the operation
\$A^{-1}\\cdot B\$ that can be implemented by external packages to take advantage of any
known structure of \$B\$.
"""
function ldiv_Bmatrix! end

"""
    MPISchurComplementBlockAinvDotB

Abstract type that can be used to indicate that a struct (defined in some other package)
is to be used as a buffer storing \$A^{-1}\\cdot B\$, and that it is guaranteed that
\$A^{-1}\\cdot B\$ has a block structure such that any row has non-zero entries only on a
single shared-memory block, so there is no need to handle 'overlaps' over distributed-MPI.
It is also assumed that the block structure corresponds to the block structure of
`A_factorization`, so that no shared-memory synchronization is needed in between copying
\$B\$ into an MPISchurComplementBlockAinvDotB and applying `A_factorization` to it (or if
synchronization is required, it is handled by the external package that defines the
methods implementing these operations).

The block structure guarantee is useful because it means that instances of
MPISchurComplementBlockAinvDotB do not have to support `getindex()` or `setindex!()`,
which would be needed to get/update the overlapping entries.

Methods `copy_B_submatrix!(Ainv_dot_B::MPISchurComplementBlockAinvDotB,
B::AbstractMatrix)`, `ldiv_Bmatrix!(A_factorization, B::MPISchurComplementBlockAinvDotB)`,
`Ainv_dot_B_dot_y!(top_vec_buffer::AbstractVector,
Ainv_dot_B::MPISchurComplementBlockAinvDotB, global_y::AbstractVector)`, and
`mul_C_Ainv_dot_B!(C_dot_Ainv_dot_B::AbstractMatrix, C::MPISchurComplementBlockC,
Ainv_dot_B::MPISchurComplementBlockAinvDotB)` must be defined by the package supplying an
implementation of MPISchurComplementBlockAinvDotB.
"""
abstract type MPISchurComplementBlockAinvDotB end

"""
    copy_B_submatrix!(Ainv_dot_B::MPISchurComplementBlockAinvDotB, B::AbstractMatrix)

Copy entries from `B` into `Ainv_dot_B`. Any entries in `Ainv_dot_B` that are not set to
values from `B` (e.g. if `B` is a sparse matrix and has some structural zeros) must also
be set to zero in this function.
"""
function copy_B_submatrix! end

"""
    Ainv_dot_B_dot_y!(top_vec_buffer::AbstractVector,
                      Ainv_dot_B::MPISchurComplementBlockAinvDotB,
                      global_y::AbstractVector)

Compute the product `Ainv_dot_B * global_y`, storing the output in `top_vec_buffer`.
"""
function Ainv_dot_B_dot_y! end

"""
    MPISchurComplementBlockC

Abstract type that can be used to indicate that a struct (defined in some other package)
is to be used as a buffer storing \$C\$, and that it is guaranteed that \$C\$ has a block
structure such that any column has non-zero entries only on a single shared-memory block,
so there is no need to handle 'overlaps' over distributed-MPI.

The block structure guarantee is useful because it means that instances of
MPISchurComplementBlockC do not have to support `getindex()` or `setindex!()`, which would
be needed to get/update the overlapping entries.

Methods `copy_C_submatrix!(C_buffer::MPISchurComplementBlockC, C::AbstractMatrix)` and
`mul_C_Ainv_dot_B!(C_dot_Ainv_dot_B::AbstractMatrix, C::MPISchurComplementBlockC,
Ainv_dot_B::MPISchurComplementBlockAinvDotB)` must be defined by the package supplying an
implementation of MPISchurComplementBlockC.
"""
abstract type MPISchurComplementBlockC end

"""
    copy_C_submatrix!(C_buffer::MPISchurComplementBlockC, C::AbstractMatrix)

Copy entries from `C` into `C_buffer`. Any entries in `C_buffer` that are not set to
values from `C` (e.g. if `C` is a sparse matrix and has some structural zeros) must also
be set to zero in this function.
"""
function copy_C_submatrix! end

"""
    mul_C_dot_Ainv_dot_u!(C_dot_Ainv_dot_u::AbstractVector, C::MPISchurComplementBlockC,
                          Ainv_dot_u::AbstractVector)

Compute the product `C * Ainv_dot_u`, storing the output in `C_dot_Ainv_dot_u`.
"""
function mul_C_dot_Ainv_dot_u! end

"""
    mul_C_Ainv_dot_B!(C_dot_Ainv_dot_B::AbstractMatrix, C::MPISchurComplementBlockC,
                      Ainv_dot_B::MPISchurComplementBlockAinvDotB)

Compute `C * Ainv_dot_B`, storing the result in `C_dot_Ainv_dot_B`.
"""
function mul_C_Ainv_dot_B! end

struct MPISchurComplement{Tf<:AbstractFloat,TA,TAiB,TAiBl,TB,
                          Tdensebuff<:Union{AbstractMatrix{Tf},Nothing},
                          TC<:Union{AbstractMatrix{Tf},MPISchurComplementBlockC},
                          TSC<:AbstractMatrix{Tf},TSCF,TAiu,TCAiB,TCAiu,TAiBy,Ttv,Tltv,
                          Tbv,Tgy,TBob,Trangeno,Tsync,Ttimer} <: Factorization{Tf}
    A_factorization::TA
    Ainv_dot_B::TAiB
    Ainv_dot_B_local::TAiBl
    B::TB
    B_column_range_partial::UnitRange{Int64}
    B_global_column_range::Trange
    B_global_column_range_partial::Trange
    B_local_column_range::Trange
    B_local_column_range_partial::Trange
    B_local_column_repeats::Matrix{Int64}
    B_local_column_repeats_partial::Matrix{Int64}
    B_dense_buffer::Tdensebuff
    C::TC
    C_global_row_range_partial::Trange
    C_local_row_range_partial::Trange
    C_local_row_repeats_partial::Matrix{Int64}
    C_row_counter::Vector{Int64}
    C_dense_buffer::Tdensebuff
    D_global_column_range_partial::Trange
    D_local_column_range_partial::Trange
    D_local_column_repeats::Matrix{Int64}
    D_local_column_repeats_partial::Matrix{Int64}
    D_dense_buffer::Tdensebuff
    schur_complement::TSC
    schur_complement_factorization::TSCF
    schur_complement_local_range_partial::Trange
    Ainv_dot_u::TAiu
    C_dot_Ainv_dot_B::TCAiB
    C_dot_Ainv_dot_u::TCAiu
    Ainv_dot_B_dot_y::TAiBy
    top_vec_buffer::Ttv
    local_top_vec_buffer::Tltv
    top_vec_local_size::Int64
    bottom_vec_buffer::Tbv
    bottom_vec_local_size::Int64
    global_y::Tgy
    top_vec_global_size::Int64
    bottom_vec_global_size::Int64
    owned_top_vector_entries::Trange
    local_top_vector_range_partial::UnitRange{Int64}
    local_top_vector_unique_entries_partial::Trange
    local_top_vector_overlaps::Vector{Trange}
    local_top_vector_repeats::Matrix{Int64}
    local_top_vector_repeats_partial::Matrix{Int64}
    B_overlap_buffers_send::TBob
    B_overlap_buffers_recv::TBob
    overlap_ranks::Vector{Int64}
    owned_bottom_vector_entries::Trange
    unique_bottom_vector_entries::Trange
    global_bottom_vector_range_partial::Trange
    global_bottom_vector_entries_no_overlap_partial::Trange
    local_bottom_vector_range_partial::UnitRange{Int64}
    local_bottom_vector_entries_no_overlap_partial::Trangeno
    local_bottom_vector_unique_entries::Vector{Int64}
    local_bottom_vector_repeats::Matrix{Int64}
    local_bottom_vector_repeats_partial::Matrix{Int64}
    comm::MPI.Comm
    shared_comm::MPI.Comm
    shared_rank::Int64
    distributed_comm::MPI.Comm
    distributed_rank::Int64
    distributed_nproc::Int64
    synchronize_shared::Tsync
    use_sparse::Bool
    separate_Ainv_B::Bool
    parallel_schur::Bool
    check_lu::Bool
    timer::Ttimer
end

"""
    remove_gaps_in_ranges!(distributed_ranges::Vector{Vector{Int64}})

Where `distributed_ranges` does not include only a contiguous range (possibly overlapping)
of indices starting from 1, identify and remove any 'gaps' (indices 1 or greater and less
than the largest index in `distributed_ranges` which are not present in any element of
`distributed_ranges`) from the index ranges.
"""
function remove_gaps_in_ranges!(distributed_ranges::Vector{Vector{Int64}})
    # This may be a bit inefficient when the state vector size gets large. In that
    # case, it is suggested to use UnitRange{Int64}-based ranges.
    all_indices = vcat(distributed_ranges...)
    sort!(all_indices)
    gaps = UnitRange{Int64}[]
    prev_ind = 0
    for i ∈ all_indices
        if i > prev_ind + 1
            push!(gaps, prev_ind+1:i-1)
        end
        prev_ind = i
    end

    if length(gaps) > 0
        # Remove `gaps` from distributed_ranges
        for this_range ∈ distributed_ranges
            igap = 1
            n_gaps = length(gaps)
            this_gap = gaps[igap]
            # Allow for the possibility that this_range is not sorted, but assume that it
            # is mostly sorted (i.e. there are a small number of times when
            # this_ind < ind_previous) - otherwise the following would be inefficient.
            ind_previous = 0
            # Start here so that if the first index is >1 (i.e. there is an
            # 'initial gap') we also remove that.
            gaps_offset = 0
            for i ∈ 1:length(this_range)
                this_ind = this_range[i]
                if this_ind ≤ ind_previous
                    # Restart the gap search because the range is not sorted.
                    igap = 1
                    this_gap = gaps[igap]
                    gaps_offset = 0
                end
                ind_previous = this_ind
                if this_ind > this_gap.start
                    while igap ≤ n_gaps && gaps[igap].stop < this_ind
                        gaps_offset += length(gaps[igap])
                        igap += 1
                    end
                    this_gap = gaps[min(igap, n_gaps)]
                end
                this_range[i] -= gaps_offset
            end
        end
    end

    return nothing
end

"""
    separate_repeated_indices(inds)

Find any repeated indices in `inds`. The first occurence of and index is the 'real' entry,
and any subsequent occurences are 'repeats. Returns:
1. a Vector of 'real' entries
2. a Vector of the positions within the vector (i.e. the local indices) of the 'real' entries
3. a 2xn Matrix containing all the repeats, where the two entries in each column are the
   'real' index and a corresponding 'repeat'
"""
function separate_repeated_indices(inds)
    indices_with_repeats = eltype(inds)[]
    unique_inds = eltype(inds)[]
    for i ∈ inds
        if i ∈ unique_inds
            push!(indices_with_repeats, i)
        else
            push!(unique_inds, i)
        end
    end
    n_repeats = length(indices_with_repeats)
    # Get rid of any repeated repeats from indices_with_repeats.
    unique!(indices_with_repeats)

    repeats = zeros(Int64, 2, n_repeats)
    counter = 0
    for repeated ∈ indices_with_repeats
        all_repeats = findall(i -> i==repeated, inds)
        first_occurence = all_repeats[1]
        # Skip the first entry in all_repeats which is the position of the 'real' entry.
        for position ∈ all_repeats[2:end]
            counter += 1
            repeats[1,counter] = first_occurence
            repeats[2,counter] = position
        end
    end
    if counter != n_repeats
        error("Filled $counter columns of `repeats`, but expected $n_repeats repeats.")
    end

    local_unique_inds = find_local_vector_inds(unique_inds, inds)

    return unique_inds, local_unique_inds, repeats
end

function get_partial_repeated_inds(repeats, partial_range)
    if repeats === nothing
        return nothing
    end
    # Filter all repeats to those where the first_occurrence i in partial_range, so that
    # the repeats can be handled using shared-memory parallelism.
    local_repeat_columns = findall(x->(x ∈ partial_range), repeats[1,:])
    partial_repeats = repeats[:,local_repeat_columns]
    return partial_repeats
end

"""
    find_local_vector_inds(global_inds, owned_global_inds)

Find indices of `global_inds` within `owned_global_inds`. This gives the 'local indices'
corresponding to `global_inds`.
"""
function find_local_vector_inds(global_inds::AbstractArray, owned_global_inds)
    local_inds = similar(global_inds)
    for (i, gind) ∈ enumerate(global_inds)
        # `owned_global_inds` is not necessarily sorted, so no obvious way to optimise
        # this.
        local_inds[i] = findfirst(i->i==gind, owned_global_inds)
    end
    return local_inds
end

"""
    update_sparse_matrix!(A::AbstractSparseMatrixCSC{Tf,Ti},
                          new_A::SparseMatrixCSC{Tf,Ti}) where {Tf,Ti}

Update the values of `A` in-place to the values of `new_A`. May not be ideally efficient
because it requires resizing Vectors.
"""
function update_sparse_matrix!(A::AbstractSparseMatrixCSC{Tf,Ti},
                               new_A::SparseMatrixCSC{Tf,Ti}) where {Tf,Ti}
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval
    new_colptr = new_A.colptr
    new_rowval = new_A.rowval
    new_nzval = new_A.nzval
    resize!(colptr, length(new_colptr))
    colptr .= new_colptr
    resize!(rowval, length(new_rowval))
    rowval .= new_rowval
    resize!(nzval, length(new_nzval))
    nzval .= new_nzval
    return nothing
end

"""
    update_sparse_matrix!(A::SparseMatrixCSR{Bi,Tf,Ti},
                          new_A::SparseMatrixCSC{Tf,Ti}, new_rowinds,
                          row_counter::Vector{Int64}) where {Bi,Tf,Ti}

Update the values of `A` in-place to the values of `new_A`. May not be ideally efficient
because it requires resizing Vectors.

`new_rowinds` gives the subset of rows in `new_A` that should be copied into `A`.

`row_counter` is an integer buffer used to help keep track of the current row in each
column of the 'compressed-sparse-column' matrix `new_A`.
"""
function update_sparse_matrix!(A::SparseMatrixCSR{Bi,Tf,Ti},
                               new_A::AbstractSparseMatrixCSC{Tf,Ti}, new_rowinds,
                               row_counter::Vector{Int64}) where {Bi,Tf,Ti}
    rowptr = A.rowptr
    colval = A.colval
    nzval = A.nzval
    new_colptr = new_A.colptr
    new_rowval = new_A.rowval
    new_nzval = new_A.nzval
    resize!(rowptr, 1)
    resize!(colval, 0)
    resize!(nzval, 0)

    if isempty(new_rowinds)
        return nothing
    end

    new_first_row = first(new_rowinds)
    for col ∈ 1:size(new_A, 2)
        new_firsti = new_colptr[col]
        new_lasti = new_colptr[col+1]-1
        col_rv = @view new_rowval[new_firsti:new_lasti]
        row_counter[col] = max(searchsortedlast(col_rv, new_first_row) - 1, 1) + new_firsti - 1
    end

    nrow, ncol = size(A)
    for row ∈ 1:nrow
        new_row = new_rowinds[row]
        for col ∈ 1:ncol
            newi = row_counter[col]
            new_lasti = new_colptr[col+1]-1
            while newi ≤ new_lasti && new_rowval[newi] < new_row
                newi += 1
            end
            if newi > new_lasti
                row_counter[col] = newi
                continue
            end
            if new_rowval[newi] == new_row
                val = new_nzval[newi]
                if val != zero(Tf)
                    push!(colval, col)
                    push!(nzval, val)
                end
            end
            row_counter[col] = newi
        end
        push!(rowptr, length(colval) + 1)
    end

    return nothing
end

"""
    update_sparse_matrix!(A::SparseMatrixCSR{Bi,Tf,Ti},
                          new_A::SparseMatrixCSC{Tf,Ti}, new_rowinds, new_colinds,
                          row_counter::Vector{Int64}) where {Bi,Tf,Ti}

Update the values of `A` in-place to the values of `new_A`. May not be ideally efficient
because it requires resizing Vectors.

`new_rowinds` gives the subset of rows in `new_A` that should be copied into `A`.

`new_colinds` gives the subset of columns in `new_A` that should be copied into `A`.

`row_counter` is an integer buffer used to help keep track of the current row in each
column of the 'compressed-sparse-column' matrix `new_A`.
"""
function update_sparse_matrix!(A::SparseMatrixCSR{Bi,Tf,Ti},
                               new_A::AbstractSparseMatrixCSC{Tf,Ti}, new_rowinds, new_colinds,
                               row_counter::Vector{Int64}) where {Bi,Tf,Ti}
    rowptr = A.rowptr
    colval = A.colval
    nzval = A.nzval
    new_colptr = new_A.colptr
    new_rowval = new_A.rowval
    new_nzval = new_A.nzval
    resize!(rowptr, 1)
    resize!(colval, 0)
    resize!(nzval, 0)

    if isempty(new_rowinds)
        # No entries to update
        return nothing
    end

    new_first_row = first(new_rowinds)
    for col ∈ 1:length(new_colinds)
        newcol = new_colinds[col]
        new_firsti = new_colptr[newcol]
        new_lasti = new_colptr[newcol+1]-1
        col_rv = @view new_rowval[new_firsti:new_lasti]
        row_counter[col] = max(searchsortedlast(col_rv, new_first_row) - 1, 1) + new_firsti - 1
    end

    nrow, ncol = size(A)
    for row ∈ 1:nrow
        new_row = new_rowinds[row]
        for col ∈ 1:ncol
            newcol = new_colinds[col]
            newi = row_counter[col]
            new_lasti = new_colptr[newcol+1]-1
            if new_rowval[newi] > new_row
                continue
            end
            while newi ≤ new_lasti && new_rowval[newi] < new_row
                newi += 1
            end
            if newi > new_lasti
                continue
            end
            if new_rowval[newi] == new_row
                val = new_nzval[newi]
                if val != zero(Tf)
                    push!(colval, col)
                    push!(nzval, val)
                end
            end
            row_counter[col] = newi
        end
        push!(rowptr, length(colval) + 1)
    end

    return nothing
end
@inline function update_sparse_matrix!(A::SparseMatrixCSR{Bi,Tf,Ti},
                                       new_A::SubArray{Tf,2}, new_rowinds,
                                       row_counter::Vector{Int64}) where {Bi,Tf,Ti}
    full_rowinds, full_colinds = new_A.indices
    return update_sparse_matrix!(A, parent(new_A), @view(full_rowinds[new_rowinds]),
                                 full_colinds, row_counter)
end

"""
    update_sparse_matrix!(A::AbstractSparseMatrixCSC{Tf,Ti},
                          new_A::FixedMatrixCSC{Tf,Ti}) where {Tf,Ti}

Update the values of `A` in-place to the values of `new_A`. May not be ideally efficient
because it requires resizing Vectors. For this FixedMatrixCSC version, also filter out
zeros because FixedMatrixCSC was probably defined with a maximal stencil, which might
contain extra zeros.
"""
function update_sparse_matrix!(A::AbstractSparseMatrixCSC{Tf,Ti},
                               new_A::FixedSparseCSC{Tf,Ti}) where {Tf,Ti}
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval
    new_colptr = new_A.colptr
    new_rowval = new_A.rowval
    new_nzval = new_A.nzval
    resize!(colptr, 0)
    resize!(rowval, 0)
    resize!(nzval, 0)
    count = 1
    for col ∈ 1:new_A.n
        push!(colptr, count)
        colstart = new_colptr[col]
        colend = new_colptr[col+1] - 1
        for new_i ∈ colstart:colend
            if !iszero(new_nzval[new_i])
                push!(rowval, new_rowval[new_i])
                push!(nzval, new_nzval[new_i])
                count += 1
            end
        end
    end
    push!(colptr, count)
    return nothing
end

"""
    update_sparse_matrix_select_columns!(A::FixedSparseCSC{Tf,Ti}, colinds,
                                         new_A::AbstractSparseMatrixCSC{Tf,Ti},
                                         new_colinds) where {Tf,Ti}

Update the values of `A` in-place to the values of `new_A`. May not be ideally efficient
because it requires resizing Vectors. For this FixedMatrixCSC version, also filter out
zeros because FixedMatrixCSC was probably defined with a maximal stencil, which might
contain extra zeros.

`new_colinds` gives the subset of columns in `new_A` that should be copied into the subset
of columns given by `colinds` in `A`.
"""
function update_sparse_matrix_select_columns!(A::FixedSparseCSC{Tf,Ti}, colinds,
                                              new_A::AbstractSparseMatrixCSC{Tf,Ti},
                                              new_colinds) where {Tf,Ti}
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval
    new_colptr = new_A.colptr
    new_rowval = new_A.rowval
    new_nzval = new_A.nzval
    for (col, new_col) ∈ zip(colinds, new_colinds)
        firsti = colptr[col]
        lasti = colptr[col+1] - 1
        new_firsti = new_colptr[new_col]
        new_lasti = new_colptr[new_col+1] - 1
        # Expect than usually the sparsity patterns of A and new_A will match, so the
        # rowval entries for this column will be the same in both. Therefore no need to
        # use `searchsortedlast()` to speed up finding the first matching entry.
        i = firsti
        for new_i ∈ new_firsti:new_lasti
            new_row = new_rowval[new_i]
            while i ≤ lasti && rowval[i] < new_row
                i += 1
            end
            if i ≤ lasti && rowval[i] == new_row
                nzval[i] = new_nzval[new_i]
                i += 1
            end
        end
    end
    return nothing
end

"""
    update_sparse_matrix_select_columns!(A::FixedSparseCSC{Tf,Ti}, colinds,
                                         new_A::AbstractSparseMatrixCSC{Tf,Ti},
                                         new_colinds, new_rowinds) where {Tf,Ti}

Update the values of `A` in-place to the values of `new_A`. May not be ideally efficient
because it requires resizing Vectors. For this FixedMatrixCSC version, also filter out
zeros because FixedMatrixCSC was probably defined with a maximal stencil, which might
contain extra zeros.

`new_colinds` gives the subset of columns in `new_A` that should be copied into the subset
of columns given by `colinds` in `A`.

`new_rowinds` gives the subset of rows in `new_A` that should be copied into `A`.
"""
function update_sparse_matrix_select_columns!(A::FixedSparseCSC{Tf,Ti}, colinds,
                                              new_A::AbstractSparseMatrixCSC{Tf,Ti},
                                              new_colinds, new_rowinds) where {Tf,Ti}
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval
    new_colptr = new_A.colptr
    new_rowval = new_A.rowval
    new_nzval = new_A.nzval
    new_nrows = length(new_rowinds)
    for (col, new_col) ∈ zip(colinds, new_colinds)
        firsti = colptr[col]
        lasti = colptr[col+1] - 1
        new_firsti = new_colptr[new_col]
        new_lasti = new_colptr[new_col+1] - 1
        new_firstrow = new_rowval[new_firsti]
        # Expect than usually the sparsity patterns of A and new_A will match, so the
        # rowval entries for this column will be the same in both. Therefore no need to
        # use `searchsortedlast()` to speed up finding the first matching entry for `i`.
        i = firsti
        row = max(searchsortedlast(new_rowinds, new_firstrow) - 1, 1)
        for new_i ∈ new_firsti:new_lasti
            new_row = new_rowval[new_i]
            while row ≤ new_nrows && new_rowinds[row] < new_row
                row += 1
            end
            if row > new_nrows
                break
            end
            if new_rowinds[row] == new_row
                while i ≤ lasti && rowval[i] < row
                    i += 1
                end
                if i ≤ lasti && rowval[i] == row
                    nzval[i] = new_nzval[new_i]
                    i += 1
                    row += 1
                end
            end
        end
    end
    return nothing
end
function update_sparse_matrix_select_columns!(A::FixedSparseCSC{Tf,Ti}, colinds,
                                              new_A::AbstractMatrix{Tf}, new_colinds,
                                              new_rowinds) where {Tf,Ti}
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval
    new_nrows = length(new_rowinds)
    for (col, new_col) ∈ zip(colinds, new_colinds)
        firsti = colptr[col]
        lasti = colptr[col+1] - 1
        for i ∈ firsti:lasti
            row = rowval[i]
            new_row = new_rowinds[row]
            nzval[i] = new_A[new_row,new_col]
        end
    end
    return nothing
end
@inline function update_sparse_matrix_select_columns!(A::FixedSparseCSC{Tf,Ti}, colinds,
                                                      new_A::SubArray{Tf,2},
                                                      new_colinds) where {Tf,Ti}
    full_new_A = parent(new_A)
    full_rowinds, full_colinds = new_A.indices
    return update_sparse_matrix_select_columns!(A, colinds, full_new_A,
                                                @view(full_colinds[new_colinds]),
                                                full_rowinds)
end

"""
    update_from_sparse_matrix_select_columns!(A::Matrix{Tf},
                                              new_A::AbstractSparseMatrixCSC{Tf,Ti},
                                              new_colinds, new_rowinds) where {Tf,Ti}

Update the values of `A` in-place to the values of `new_A`.

`new_colinds` gives the subset of columns in `new_A` that should be copied into `A`.

`new_rowinds` gives the subset of rows in `new_A` that should be copied into `A`.
"""
function update_from_sparse_matrix_select_columns!(A::Matrix{Tf}, colinds,
                                                   new_A::AbstractSparseMatrixCSC{Tf,Ti},
                                                   new_colinds, new_rowinds) where {Tf,Ti}
    new_colptr = new_A.colptr
    new_rowval = new_A.rowval
    new_nzval = new_A.nzval
    new_nrows = length(new_rowinds)
    for (col, new_col) ∈ zip(colinds, new_colinds)
        new_firsti = new_colptr[new_col]
        new_lasti = new_colptr[new_col+1] - 1
        new_firstrow = new_rowval[new_firsti]
        # Expect than usually the sparsity patterns of A and new_A will match, so the
        # rowval entries for this column will be the same in both. Therefore no need to
        # use `searchsortedlast()` to speed up finding the first matching entry for `i`.
        row = max(searchsortedlast(new_rowinds, new_firstrow) - 1, 1)
        A[1:row-1,col] .= 0.0
        for new_i ∈ new_firsti:new_lasti
            new_row = new_rowval[new_i]
            while row ≤ new_nrows && new_rowinds[row] < new_row
                A[row,col] = 0.0
                row += 1
            end
            if row > new_nrows
                A[row:end,col] .= 0.0
                break
            end
            if new_rowinds[row] == new_row
                A[row,col] = new_nzval[new_i]
                row += 1
            end
        end
        A[row:end,col] .= 0.0
    end
    return nothing
end
@inline function update_from_sparse_matrix_select_columns!(A::Matrix{Tf}, colinds,
                                                           new_A::SubArray{Tf,2},
                                                           new_colinds) where {Tf}
    full_new_A = parent(new_A)
    full_rowinds, full_colinds = new_A.indices
    return update_from_sparse_matrix_select_columns!(A, colinds, full_new_A,
                                                     @view(full_colinds[new_colinds]),
                                                     full_rowinds)
end

"""
    update_sparse_matrix_select_rows!(A::AbstractSparseMatrixCSC{Tf,Ti},
                                      new_A::FixedSparseCSC{Tf,Ti}, rowinds) where {Tf,Ti}

Update the values of `A` in-place to the values of `new_A`. May not be ideally efficient
because it requires resizing Vectors. For this FixedMatrixCSC version, also filter out
zeros because FixedMatrixCSC was probably defined with a maximal stencil, which might
contain extra zeros.

`rowinds` gives the subset of rows in `new_A` that should be copied into `A`.
"""
function update_sparse_matrix_select_rows!(A::AbstractSparseMatrixCSC{Tf,Ti},
                                           new_A::FixedSparseCSC{Tf,Ti}, rowinds) where {Tf,Ti}
    colptr = A.colptr
    rowval = A.rowval
    nzval = A.nzval
    new_colptr = new_A.colptr
    new_rowval = new_A.rowval
    new_nzval = new_A.nzval
    resize!(colptr, 0)
    resize!(rowval, 0)
    resize!(nzval, 0)
    count = 1
    n_rowinds = length(rowinds)
    for col ∈ 1:new_A.n
        push!(colptr, count)
        colstart = new_colptr[col]
        colend = new_colptr[col+1] - 1
        if colend < colstart
            continue
        end
        row_count = max(searchsortedlast(rowinds, new_rowval[colstart]) - 1, 1)
        for new_i ∈ colstart:colend
            rv = new_rowval[new_i]
            while row_count ≤ n_rowinds && rowinds[row_count] < rv
                row_count += 1
            end
            if row_count > n_rowinds
                break
            end
            if rowinds[row_count] == rv
                newval = new_nzval[new_i]
                if !iszero(newval)
                    push!(rowval, row_count)
                    push!(nzval, newval)
                    count += 1
                    row_count += 1
                end
            end
        end
    end
    push!(colptr, count)
    return nothing
end

function get_partial_FixedSparseCSC_buffer(row_range, existing_buffer, data_type)
    # Initialize buffer with the same non-zero pattern as existing_buffer, but only for a
    # subset of rows given by row_range.
    ncol = size(existing_buffer, 2)
    if isempty(row_range)
        return FixedSparseCSC(0, ncol, ones(Int64, ncol + 1), Int64[], zeros(data_type, 0))
    end
    colptr = Int64[1]
    rowval = Int64[]
    firstrow = first(row_range)
    lastrow = last(row_range)
    existing_colptr = existing_buffer.colptr
    existing_rowval = existing_buffer.rowval
    for j ∈ 1:ncol
        existing_col_start = existing_colptr[j]
        existing_col_end = existing_colptr[j+1]-1
        existing_col_rowval = @view existing_rowval[existing_col_start:existing_col_end]
        n_existing = existing_col_end - existing_col_start + 1
        if n_existing == 0 || first(existing_col_rowval) > lastrow || last(existing_col_rowval) < firstrow
            # Definitely no overlapping entries in this column, so skip.
            push!(colptr, length(rowval) + 1)
            continue
        end
        count = max(searchsortedlast(existing_col_rowval, firstrow) - 1, 1)
        for (i, i_global) ∈ enumerate(row_range)
            while count ≤ n_existing && existing_col_rowval[count] < i_global
                count += 1
            end
            if count > n_existing
                break
            end
            if existing_col_rowval[count] == i_global
                push!(rowval, i)
            end
        end
        push!(colptr, length(rowval) + 1)
    end
    nzval = zeros(data_type, length(rowval))
    buffer = FixedSparseCSC(length(row_range), ncol, colptr, rowval, nzval)
    return buffer
end

# SparseMatrixCSR multiplying SparseMatrixCSC currently (26/5/2026) uses the generic
# AbstractMatrix implementation, which is very slow. The following is a more optimised
# implementation. Adapted from `SparseArrays._spmatmul!(C, A, B, α, β)`.
# SparseArrays.jl license:
### MIT License
###
### Copyright (c) 2018-2024 SparseArrays.jl contributors: https://github.com/JuliaSparse/SparseArrays.jl/contributors
###
### Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
###
### The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
###
### THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# Slow non-inlined functions for throwing the error without messing up the caller
@noinline function _matmul_size_error(mC, nC, mA, nA, mB, nB, At, Bt)
    if At == 'N'
        Anames = "first", "second"
    else
        Anames = "second", "first"
    end
    if Bt == 'N'
        Bnames = "first", "second"
    else
        Bnames = "second", "first"
    end
    nA == mB ||
        throw(DimensionMismatch("$(Anames[2]) dimension of A, $nA, does not match the $(Bnames[1]) dimension of B, $mB"))
    mA == mC ||
        throw(DimensionMismatch("$(Anames[1]) dimension of A, $mA, does not match the first dimension of C, $mC"))
    nB == nC ||
        throw(DimensionMismatch("$(Bnames[2]) dimension of B, $nB, does not match the second dimension of C, $nC"))
    # unreachable
    throw(DimensionMismatch("Unknown dimension mismatch"))
end
# `_matmul_size_AB()`, that SparseArrays uses in _spmul!(), is an internal function in
# SparseArrays which might change in future, so just copy a slightly simplified version
# here to avoid having to follow any changes.
@inline function _matmul_size(C, A, B)
    mC = size(C, 1)
    nC = size(C, 2)
    mA = size(A, 1)
    nA = size(A, 2)
    mB = size(B, 1)
    nB = size(B, 2)

    if (nA != mB) | (mA != mC) | (nB != nC)
        _matmul_size_error(mC, nC, mA, nA, mB, nB, At, Bt)
    end
    return mC, nC, mA, nA, mB, nB
end
function csr_mul!(C::AbstractSparseMatrixCSC{Tf}, A::SparseMatrixCSR{Bi,Tf},
                  B::AbstractSparseMatrixCSC{Tf}, α::Number, β::Number) where {Bi,Tf}
    if Bi != 1
        error("Only 1-based indexing supported here")
    end
    Cax2 = axes(C, 2)
    Aax1 = axes(A, 1)
    mC, nC, mA, nA, mB, nB = _matmul_size(C, A, B)
    nzv = nonzeros(A)
    cv = colvals(A)
    rp = A.rowptr
    isone(β) || LinearAlgebra._rmul_or_fill!(C, β)
    if α isa Bool && !α
        return
    end
    Bcp = B.colptr
    Brv = rowvals(B)
    Bnz = nonzeros(B)
    Ccp = C.colptr
    Crv = rowvals(C)
    Cnz = nonzeros(C)
    @inbounds for k in Cax2
        B_flat_start = Bcp[k]
        B_flat_end = Bcp[k+1]-1
        Bcol_rv = @view Brv[B_flat_start:B_flat_end]
        Bcol_nz = @view Bnz[B_flat_start:B_flat_end]
        nb = B_flat_end - B_flat_start + 1
        C_flat_start = Ccp[k]
        C_flat_end = Ccp[k+1]-1
        Ccol_rv = @view Crv[C_flat_start:C_flat_end]
        Ccol_nz = @view Cnz[C_flat_start:C_flat_end]
        nc = C_flat_end - C_flat_start + 1
        C_count = max(searchsortedlast(Ccol_rv, first(Aax1)) - 1, 1)
        for row in Aax1
            temp = zero(Tf)
            firstj = rp[row]
            lastj = rp[row+1]-1
            if lastj < firstj
                # No entries on this row.
                continue
            end
            B_count = max(searchsortedlast(Bcol_rv, cv[firstj]) - 1, 1)
            for j in firstj:lastj
                cvj = cv[j]
                while B_count ≤ nb && Bcol_rv[B_count] < cvj
                    B_count += 1
                end
                if B_count ≤ nb && Bcol_rv[B_count] == cvj
                    temp = muladd(nzv[j], Bcol_nz[B_count], temp)
                end
            end
            temp = α isa Bool ? temp : temp * α
            if temp != zero(Tf)
                if isa(C, FixedSparseCSC)
                    # Non-zero pattern must already be set.
                    while Ccol_rv[C_count] < row
                        C_count += 1
                    end
                    if Ccol_rv[C_count] != row
                        error("Attempting to insert into structural zero of C.")
                    end
                    Ccol_nz[C_count] = temp
                else
                    C[row,k] = temp
                end
            end
        end
    end
    return C
end

"""
    mpi_schur_complement(A_factorization, B::Union{AbstractMatrix,Nothing,Type},
                         C::Union{AbstractMatrix,Nothing,Type},
                         D::Union{AbstractMatrix,Nothing,Type},
                         owned_top_vector_entries::Union{UnitRange{Int64},Vector{Int64}},
                         owned_bottom_vector_entries::Union{UnitRange{Int64},Vector{Int64}};
                         B_global_column_range::Union{UnitRange{Int64},Vector{Int64},Nothing}=nothing,
                         C_global_row_range::Union{UnitRange{Int64},Vector{Int64},Nothing}=nothing,
                         D_global_column_range::Union{UnitRange{Int64},Vector{Int64},Nothing}=nothing,
                         comm::MPI.Comm=MPI.COMM_WORLD,
                         shared_comm::MPI.Comm=MPI.COMM_SELF,
                         distributed_comm::Union{MPI.Comm,Nothing}=nothing,
                         allocate_shared_float::Union{Function,Nothing}=nothing,
                         allocate_shared_int::Union{Function,Nothing}=nothing,
                         synchronize_shared::Union{Function,Nothing}=nothing,
                         use_sparse::Bool=true, separate_Ainv_B::Bool=false,
                         sparse_Ainv_B::Bool=false,
                         parallel_schur::Union{Bool,Factorization}=(distributed_comm!==nothing || shared_comm!==nothing),
                         Ainv_dot_B_buffer::Union{AbstractMatrix,MPISchurComplementBlockAinvDotB,Nothing}=nothing,
                         C_buffer::Union{MPISchurComplementBlockC,Nothing}=nothing,
                         C_dot_Ainv_dot_B_buffer_ncopies::Union{Integer,Nothing}=nothing,
                         schur_complement_buffer::Union{AbstractMatrix,Nothing}=nothing,
                         copy_input_to_dense_buffers::Bool=false,
                         schur_tile_size::Union{Integer,Nothing}=nothing,
                         skip_factorization::Bool=false, check_lu::Bool=true,
                         timer::Union{TimerOutput,Nothing}=nothing)

Initialise an MPISchurComplement struct representing a 2x2 block-structured matrix
```math
\\left(\\begin{array}{cc}
A & B\\\\
C & D
\\end{array}\\right)
```

`A_factorization` should be a matrix factorization of `A`, or an object with similar
functionality, that can be passed to the `ldiv!()` function to solve a matrix system.

`B` and `D` are only used to initialize the MPISchurComplement, they are not stored. If
`sparse=false` is passed, a reference to `C` is stored. Only the locally owned parts of
`B`, `C` and `D` should be passed. `B`, `C` and `D` may be modified by this function.

`owned_top_vector_entries` gives the range of global indices that are owned by this
process in the top block of the state vector, `owned_bottom_vector_entries` the same for
the bottom block. The index ranges in `owned_top_vector_entries` and
`owned_bottom_vector_entries` may have gaps, not being a contiguous range starting from 1
(and similarly for `B_global_column_range`, `C_global_row_range` and
`D_global_column_range`, if they are passed, although they should all contain the same set
of indices, across all processes, as `owned_bottom_vector_entries`). This may be useful if
the 'bottom block' is actually, for example, a chunk from the middle of a grid, which
separates the 'top block' into block-diagonal pieces. The global indexing for the full
state vector can be used for both `owned_top_vector_entries` and
`owned_bottom_vector_entries`, and they will be converted (by shifting indices down to
remove the gaps) into a representation internal to the `MPISchurComplement` object that is
contiguous and starts at 1 for each block. In addition, the indices passed do not have to
be unique (they can be repeated on different subdomains, and/or within a single subdomain)
or monotonically increasing - this allows for periodic dimensions where the final point in
the dimension should be identified with the first point. When there are repeated indices,
the vector entries passed for the right-hand-side (`u` and `v`) are assumed to be
identical at the locations given corresponding to repeated indices. Matrix entries in `B`,
`C`, and `D` corresponding to each repeated row or column index will be (in effect) summed
together into a single entry of an 'assembled' matrix (as an implementation detail, the
'assembled' matrix may not be constructed explicitly, but it serves to define the meaning
of repeated matrix entries).

`B` should contain only the rows corresponding to `owned_top_vector_entries` and by
default the columns corresponding to `owned_bottom_vector_entries`. If a different range
of columns should be included, the corresponding global indices can be passed as
`B_global_column_range`.

`C` should contain only the columns corresponding to `owned_top_vector_entries` and by
default the rows corresponding to `owned_bottom_vector_entries`. If a different range of
rows should be included, the corresponding global indices can be passed as
`C_global_row_range`.

`D` should contain only the rows corresponding to `owned_bottom_vector_entries` and by
default the same columns. If a different range of columns should be included, the
corresponding global indices passed as `D_global_column_range`.

When shared-memory MPI parallelism is used, `B`, `C`, and `D` should all be shared memory
arrays which are identical on all processes in `shared_comm` (or non-shared identical
copies, but that would be an inefficient use of memory). When `owned_top_vector_entries`
and/or `owned_bottom_vector_entries` are ranges that overlap between different
distributed-MPI ranks, the matrices `A`, `B`, `C`, and `D` should be set up so that adding
together the matrices passed on each distributed-MPI rank gives the full, global matrix
(i.e. there should be no double-counting of overlapping entries, but any fraction of the
overlap can be passed on any distributed rank, as long as the contributions from each
distributed rank add up to the full value).

`comm` is the MPI communicator containing all processes to be used by MPIDenseLU.

`shared_comm` is the MPI communicator to use for shared-memory communications.

`distributed_comm` (if passed) is the MPI communicator for distributed-memory
communications. It should include only the ranks that are rank-0 in each `shared_comm`.
By default it will be created from `comm`, but can be passed in to avoid creating extra
`MPI.Comm`s.

`allocate_shared_float` and `allocate_shared_int` must be passed when `shared_comm` is
passed. They should be passed functions that will be used to allocate various buffer
arrays. `allocate_shared_float` should return arrays with the same element type as
`A_factorization`, `B`, `C`, and `D`, while `allocate_shared_int` should return arrays
with element type `Int64`. This is necessary as the MPI shared-memory 'windows'
(`MPI.Win`) have to be stored and (eventually) freed. The 'windows' must be managed
externally, as if they are freed by garbage collection, the freeing is not synchronized
between different processes, causing errors.

`synchronize_shared` can be passed a function that synchronizes all processes in
`shared_comm`. This may be useful if a custom synchronization is required for debugging,
etc. If it is not passed, `MPI.Barrier(shared_comm)` will be used.

By default, makes `C` a sparse matrix. Pass `use_sparse=false` to force `C` to be dense.

If `B` is very sparse, and `A_factorization` is efficiently parallelised, it might be more
efficient to compute `A \\ (B.y)` in two steps (`B.y` then `A \\ ()`) rather than
multiplying by the dense `Ainv_dot_B`. If `separate_Ainv_B=true` is passed, this will be
done (requires `use_sparse=true`). There is no saving in setup time because `Ainv_dot_B`
still has to be calculated, to be multiplied by `C` when calculating `schur_complement`.
There is also no memory saving as a dense B-sized buffer array is needed.

When `separate_Ainv_B=false`, `sparse_Ainv_B=true` can be passed to use sparse matrix
storage for `Ainv_dot_B`.

By default, when `distributed_comm` and/or`shared_comm` are passed the MPIDenseLUs package
is used to factorize/solve the Schur complement matrix (which is dense) using a hybrid
distributed+shared-memory MPI parallel implementation (both factorization `lu!()` and
solve `ldiv!()` phases are parallelised) and when neither communicator is is passed the
serial implementation from LinearAlgebra (which uses LAPACK/BLAS) is used.
`parallel_schur` can be passed to force use of the serial (`false`) or parallel
MPIDenseLUs (`true`) implementations. When using MPIDenseLUs, `schur_tile_size` can be
used to set the `tile_size` argument to `mpi_dense_lu()`; the default is to use the
smaller of 128 and the largest 2^n smaller than half of the size of the global 'bottom
vector'. Alternatively, a `Factorization` instance can be passed to `parallel_schur`, and
this will be used to factorize (with `lu!()`) and solve (with `ldiv!()`) the Schur
complement matrix.

To use a non-dense matrix type for `Ainv_dot_B` or `schur_complement`, pass the buffer(s)
to use as `Ainv_dot_B_buffer` or `schur_complement_buffer`. If you are using shared-memory
parallelism, these must be shared-memory buffers.

To use some custom operations (presumably with more optimized implementations that exploit
some known block structure of the sub-matrices) for the application of \$A^{-1}\$ to \$B\$
and multiplication \$C\\cdot(A^{-1}\\cdotB)\$ buffers of type
`MPISchurComplementBlockAinvDotB` and `MPISchurComplementBlockC` can be passed to
`Ainv_dot_B_buffer` and `C_buffer`. If either is passed, both must be.
`C_dot_Ainv_dot_B_buffer_ncopies` gives the number of duplicate buffers that are needed in
the shared-memory array (`C_dot_Ainv_dot_B`) that mul_C_Ainv_dot_B!() writes output to.

When the inputs that will be passed to `update_schur_complement!()` are sparse arrays (or
views of sparse arrays), and there are repeated indices, it might be the case that the
'copy to' locations of the repeated indices are not part of the non-zero entries in the
sparse matrix. In that case it is necessary to copy the `B`, `C` and `D` inputs into
buffers that have non-zero entries in all the required places. One way to do this is to
use dense-matrix buffers, and this will be done if `copy_input_to_dense_buffers=true` is
passed.

`skip_factorization=true` can be passed to create an MPISchurComplement instance without
calculating the factorization corresponding to the input matrices. `ldiv!()` called with
this instance will give incorrect results unless `update_schur_complement!()` is called
first. In this case, `B`, `C`, and `D` can be passed `nothing` or an element type instead
of matrices as the matrix values will not be used.

`check_lu=false` can be passed to disable checks when performing dense LU factorizations.
This may increase the speed of the factorization, but leaves it up to the user to
guarantee correctness of the input matrices.

A `TimerOutput` instance can be passed to `timer` to record timings of various
subroutines.
"""
function mpi_schur_complement(A_factorization, B::Union{AbstractMatrix,Nothing,Type},
                              C::Union{AbstractMatrix,Nothing,Type},
                              D::Union{AbstractMatrix,Nothing,Type},
                              owned_top_vector_entries::Union{UnitRange{Int64},Vector{Int64}},
                              owned_bottom_vector_entries::Union{UnitRange{Int64},Vector{Int64}};
                              B_global_column_range::Union{UnitRange{Int64},Vector{Int64},Nothing}=nothing,
                              C_global_row_range::Union{UnitRange{Int64},Vector{Int64},Nothing}=nothing,
                              D_global_column_range::Union{UnitRange{Int64},Vector{Int64},Nothing}=nothing,
                              comm::MPI.Comm=MPI.COMM_WORLD,
                              shared_comm::MPI.Comm=MPI.COMM_SELF,
                              distributed_comm::Union{MPI.Comm,Nothing}=nothing,
                              allocate_shared_float::Union{Function,Nothing}=nothing,
                              allocate_shared_int::Union{Function,Nothing}=nothing,
                              synchronize_shared::Union{Function,Nothing}=nothing,
                              use_sparse::Bool=true, separate_Ainv_B::Bool=false,
                              sparse_Ainv_B::Bool=false,
                              parallel_schur::Union{Bool,Factorization}=(distributed_comm!==nothing || shared_comm!==nothing),
                              Ainv_dot_B_buffer::Union{AbstractMatrix,MPISchurComplementBlockAinvDotB,Nothing}=nothing,
                              C_buffer::Union{MPISchurComplementBlockC,Nothing}=nothing,
                              C_dot_Ainv_dot_B_buffer_ncopies::Union{Integer,Nothing}=nothing,
                              schur_complement_buffer::Union{AbstractMatrix,Nothing}=nothing,
                              copy_input_to_dense_buffers::Bool=false,
                              schur_tile_size::Union{Integer,Nothing}=nothing,
                              skip_factorization::Bool=false, check_lu::Bool=true,
                              timer::Union{TimerOutput,Nothing}=nothing)

    if !skip_factorization
        if !(isa(B, AbstractMatrix) && isa(C, AbstractMatrix) && isa(D, AbstractMatrix))
            error("When `skip_factorization=false`, matrices must be passed for `B`, "
                  * "`C`, and `D`.")
        end
        data_type = eltype(D)
    elseif isa(D, Type)
        data_type = D
    elseif isa(B, Type)
        data_type = B
    elseif isa(C, Type)
        data_type = C
    else
        data_type = Float64
    end

    # Simpler to only support one type (`Vector{Int64}`) for ranges, so convert
    # UnitRange inputs to Vector.
    if isa(owned_top_vector_entries, UnitRange)
        owned_top_vector_entries = collect(owned_top_vector_entries)
    end
    if isa(owned_bottom_vector_entries, UnitRange)
        owned_bottom_vector_entries = collect(owned_bottom_vector_entries)
    end

    shared_nproc = MPI.Comm_size(shared_comm)
    shared_rank = MPI.Comm_rank(shared_comm)

    if distributed_comm == nothing
        color = shared_rank == 0 ? 0 : nothing
        distributed_comm = MPI.Comm_split(comm, color, 0)
    end

    if distributed_comm != MPI.COMM_NULL
        distributed_nproc = MPI.Comm_size(distributed_comm)
        distributed_rank = MPI.Comm_rank(distributed_comm)
    else
        distributed_nproc = -1
        distributed_rank = -1
    end

    top_vec_local_size = length(owned_top_vector_entries)
    bottom_vec_local_size = length(owned_bottom_vector_entries)
    if shared_rank == 0
        # Collect the row/column indices from all distributed ranks, and return as a
        # Vector{Trange}.
        function get_distributed_ranges(local_range::Vector{Int64})
            vec_sizes = MPI.Allgather(Cint(length(local_range)), distributed_comm)
            distributed_ranges_vec = similar(local_range, sum(vec_sizes))
            MPI.Allgatherv!(local_range, MPI.VBuffer(distributed_ranges_vec, vec_sizes),
                            distributed_comm)
            distributed_ranges = Vector{Int64}[]
            imin = 1
            for idist ∈ 1:distributed_nproc
                imax = imin + vec_sizes[idist] - 1
                push!(distributed_ranges, distributed_ranges_vec[imin:imax])
                imin = imax + 1
            end
            return distributed_ranges
        end

        top_vector_distributed_ranges = get_distributed_ranges(owned_top_vector_entries)
        bottom_vector_distributed_ranges = get_distributed_ranges(owned_bottom_vector_entries)

        remove_gaps_in_ranges!(top_vector_distributed_ranges)
        owned_top_vector_entries = top_vector_distributed_ranges[distributed_rank+1]
        remove_gaps_in_ranges!(bottom_vector_distributed_ranges)
        owned_bottom_vector_entries = bottom_vector_distributed_ranges[distributed_rank+1]

        # Include `init=0` in the `maximum()` calls in case any range is empty.
        top_vec_global_size = maximum(maximum(r; init=0)
                                      for r ∈ top_vector_distributed_ranges; init=0)
        bottom_vec_global_size = maximum(maximum(r; init=0)
                                         for r ∈ bottom_vector_distributed_ranges; init=0)

        # Find all overlaps (i.e. intersections) between locally-owned range and
        # remotely-owned ranges on all other processes.
        function get_overlaps(local_range::T, distributed_ranges) where T
            overlaps = T[]
            for idist ∈ 1:distributed_nproc
                # Remember distributed_rank is 0-based index, but idist is 1-based.
                if idist - 1 == distributed_rank || distributed_comm == MPI.COMM_NULL
                    # Don't care about overlap of this rank with itself.
                    # Make start/stop negative so they are easy to filter out later.
                    push!(overlaps, -1:-2)
                elseif idist - 1 < distributed_rank
                    # Handle idist-1<distributed_rank and idist-1>distributed_rank
                    # separately because we want to ensure a consistent orderding of the
                    # intersection indices on all processes. `intersect` maintains the
                    # order of indices, but we also need to pass the sets in the same
                    # order. If we always pass the lower rank's indices first, we will get
                    # the same output on both processes involved in the overlap.
                    push!(overlaps, intersect(distributed_ranges[idist], local_range))
                else
                    push!(overlaps, intersect(local_range, distributed_ranges[idist]))
                end
            end
            return overlaps
        end

        # For 'bottom vector' need to work out a set of non-overlapping ranges so that
        # each distributed rank owns a unique set of entries.
        bottom_vector_overlaps = get_overlaps(owned_bottom_vector_entries,
                                              bottom_vector_distributed_ranges)
        owned_bottom_vector_entries_no_overlap = copy(owned_bottom_vector_entries)
        # Say that this process 'owns' an entry if there is no process with a lower rank
        # that also owns that entry.
        # Note distributed_rank is a 0-based index, but idist is 1-based.
        for idist ∈ 1:distributed_rank
            other_proc_overlap = bottom_vector_overlaps[idist]
            filter!((i) -> i ∉ other_proc_overlap, owned_bottom_vector_entries_no_overlap)
        end
        # Filter any repeated entries out of owned_bottom_vector_entries_no_overlap.
        unique!(owned_bottom_vector_entries_no_overlap)
        unique_bottom_vector_entries, local_bottom_vector_unique_entries,
        local_bottom_vector_repeats =
            separate_repeated_indices(owned_bottom_vector_entries)

        # For any ranks that have a 'top vector overlap' that is not empty, need to create
        # a communicator so that the different ranks can sum-reduce their entries of the
        # `B` matrix.
        all_top_vector_overlaps = get_overlaps(owned_top_vector_entries,
                                               top_vector_distributed_ranges)
        local_top_vector_overlaps = Trange[]
        B_overlap_buffers_send = Matrix{data_type}[]
        B_overlap_buffers_recv = Matrix{data_type}[]
        # Not sure overlap_ranks need to be sorted, but probably nicer if they are.
        sorted_overlap_ranks = sortperm(collect(length(o) == 0 ? -1 : first(o) for o ∈ all_top_vector_overlaps))
        # Filter out empty overlaps.
        filter!((i) -> length(all_top_vector_overlaps[i]) > 0, sorted_overlap_ranks)
        # MPI ranks are given by 0-based index, but sorted_overlap_ranks are 1-based
        overlap_ranks = sorted_overlap_ranks .- 1
        for idist ∈ sorted_overlap_ranks
            this_overlap = all_top_vector_overlaps[idist]
            this_local_overlap = find_local_vector_inds(this_overlap, owned_top_vector_entries)
            push!(local_top_vector_overlaps, this_local_overlap)
            push!(B_overlap_buffers_send, zeros(data_type, length(this_overlap),
                                                bottom_vec_global_size))
            push!(B_overlap_buffers_recv, zeros(data_type, length(this_overlap),
                                                bottom_vec_global_size))
        end
        unique_top_vector_entries, local_top_vector_unique_entries,
        local_top_vector_repeats =
            separate_repeated_indices(owned_top_vector_entries)
    else
        top_vec_global_size = nothing
        bottom_vec_global_size = nothing
        unique_top_vector_entries = nothing
        local_top_vector_unique_entries = nothing
        local_top_vector_overlaps = Trange[]
        local_top_vector_repeats = nothing
        overlap_ranks = Int64[]
        B_overlap_buffers_send = nothing
        B_overlap_buffers_recv = nothing
        owned_bottom_vector_entries_no_overlap = nothing
        unique_bottom_vector_entries = nothing
        local_bottom_vector_unique_entries = nothing
        local_bottom_vector_repeats = nothing
    end

    # Communicate index ranges, etc. to all processes on the shared-memory
    # communicator.
    function shared_broadcast_int(i::Union{Int64,Nothing})
        if shared_rank == 0
            if i === nothing
                error("`i` should not be `nothing` on root of shared-memory communicator")
            end
            MPI.Bcast(i, 0, shared_comm)
            return i
        else
            return MPI.Bcast(-1, 0, shared_comm)
        end
    end
    function shared_broadcast_range(r::Union{Trange,Nothing})
        if shared_rank == 0
            if r === nothing
                error("`r` should not be `nothing` on root of shared-memory communicator")
            end
            MPI.bcast(r, 0, shared_comm)
            return r
        else
            range = MPI.bcast(Int64[], 0, shared_comm)
            return range
        end
    end
    function shared_broadcast_matrix(m::Union{Matrix{Int64},Nothing})
        if shared_rank == 0
            if m === nothing
                error("`m` should not be `nothing` on root of shared-memory communicator")
            end
            MPI.bcast(m, 0, shared_comm)
            return m
        else
            mat = MPI.bcast(zeros(Int64, 0, 0), 0, shared_comm)
            return mat
        end
    end

    distributed_rank = shared_broadcast_int(distributed_rank)
    distributed_nproc = shared_broadcast_int(distributed_nproc)
    top_vec_global_size = shared_broadcast_int(top_vec_global_size)
    local_top_vector_repeats = shared_broadcast_matrix(local_top_vector_repeats)
    bottom_vec_global_size = shared_broadcast_int(bottom_vec_global_size)
    owned_top_vector_entries = shared_broadcast_range(owned_top_vector_entries)
    unique_top_vector_entries = shared_broadcast_range(unique_top_vector_entries)
    local_top_vector_unique_entries = shared_broadcast_range(local_top_vector_unique_entries)
    owned_bottom_vector_entries = shared_broadcast_range(owned_bottom_vector_entries)
    owned_bottom_vector_entries_no_overlap = shared_broadcast_range(owned_bottom_vector_entries_no_overlap)
    unique_bottom_vector_entries = shared_broadcast_range(unique_bottom_vector_entries)
    local_bottom_vector_unique_entries = shared_broadcast_range(local_bottom_vector_unique_entries)
    local_bottom_vector_repeats = shared_broadcast_matrix(local_bottom_vector_repeats)

    # If Matrix ranges were not passed explicitly, set them from the top/bottom vector
    # ranges.
    if B_global_column_range === nothing
        B_global_column_range = unique_bottom_vector_entries
        B_local_column_range = local_bottom_vector_unique_entries
        B_local_column_repeats = local_bottom_vector_repeats
    else
        if isa(B_global_column_range, UnitRange)
            B_global_column_range = collect(B_global_column_range)
        end
        if shared_rank == 0
            B_distributed_ranges = get_distributed_ranges(B_global_column_range)
            remove_gaps_in_ranges!(B_distributed_ranges)
            B_global_column_range = B_distributed_ranges[distributed_rank+1]
        else
            B_global_column_range = Int64[]
        end
        B_global_column_range = shared_broadcast_range(B_global_column_range)
        B_global_column_range, B_local_column_range, B_local_column_repeats =
            separate_repeated_indices(B_global_column_range)
    end
    if C_global_row_range === nothing
        C_global_row_range = unique_bottom_vector_entries
        C_local_row_range = local_bottom_vector_unique_entries
        C_local_row_repeats = local_bottom_vector_repeats
    else
        if isa(C_global_row_range, UnitRange)
            C_global_row_range = collect(C_global_row_range)
        end
        if shared_rank == 0
            C_distributed_ranges = get_distributed_ranges(C_global_row_range)
            remove_gaps_in_ranges!(C_distributed_ranges)
            C_global_column_range = C_distributed_ranges[distributed_rank+1]
        else
            C_global_column_range = Int64[]
        end
        C_global_column_range = shared_broadcast_range(C_global_column_range)
        C_global_row_range, C_local_row_range, C_local_row_repeats =
            separate_repeated_indices(C_global_row_range)
    end
    if D_global_column_range === nothing
        D_global_column_range = unique_bottom_vector_entries
        D_local_column_range = local_bottom_vector_unique_entries
        D_local_column_repeats = local_bottom_vector_repeats
    else
        if isa(D_global_column_range, UnitRange)
            D_global_column_range = collect(D_global_column_range)
        end
        if shared_rank == 0
            D_distributed_ranges = get_distributed_ranges(D_global_column_range)
            remove_gaps_in_ranges!(D_distributed_ranges)
            D_global_column_range = D_distributed_ranges[distributed_rank+1]
        else
            D_global_column_range = Int64[]
        end
        D_global_column_range = shared_broadcast_range(D_global_column_range)
        D_global_column_range, D_local_column_range, D_local_column_repeats =
            separate_repeated_indices(D_global_column_range)
    end

    @boundscheck !isa(B, AbstractMatrix) || size(B, 1) == top_vec_local_size || error(BoundsError, " Rows in B do not match locally-owned 'top vector' entries.")
    @boundscheck !isa(B, AbstractMatrix) || size(B, 2) == length(B_local_column_range) + size(B_local_column_repeats, 2) || error(BoundsError, " Columns in B do not match index ranges.")
    @boundscheck !isa(C, AbstractMatrix) || size(C, 1) == length(C_local_row_range) + size(C_local_row_repeats, 2) || error(BoundsError, " Rows in C do not match index ranges.")
    @boundscheck !isa(C, AbstractMatrix) || size(C, 2) == top_vec_local_size || error(BoundsError, " Columns in C do not match locally-owned 'top vector' entries.")
    @boundscheck !isa(D, AbstractMatrix) || size(D, 1) == bottom_vec_local_size || error(BoundsError, " Rows in D do not match locally-owned 'bottom vector' entries.")
    @boundscheck !isa(D, AbstractMatrix) || size(D, 2) == length(D_local_column_range) + size(D_local_column_repeats, 2) || error(BoundsError, " Columns in D do not match index ranges.")

    if shared_comm != MPI.COMM_SELF && (allocate_shared_float === nothing
                                        || allocate_shared_int === nothing)
        error("when `shared_comm` is passed, `allocate_shared_float` and "
              * "`allocate_shared_int` arguments are required, because it is necessary "
              * "to manage the MPI.Win objects to ensure they are not garbage collected, "
              * "because garbage collection is not necessarily synchronized between "
              * "different processes.")
    end
    if allocate_shared_float === nothing
        allocate_shared_float = (args...) -> zeros(data_type, args...)
    end
    if allocate_shared_int === nothing
        allocate_shared_int = (args...) -> zeros(Int64, args...)
    end

    # Define indices that will be handled by this process in shared-memory-parallelised
    # operations.
    if shared_comm == MPI.COMM_SELF
        synchronize_shared = ()->nothing
        B_column_range_partial = 1:bottom_vec_global_size
        B_global_column_range_partial = B_global_column_range
        B_local_column_range_partial = B_local_column_range
        C_global_row_range_partial= C_global_row_range
        C_local_row_range_partial = C_local_row_range
        D_global_column_range_partial = D_global_column_range
        D_local_column_range_partial = D_local_column_range
        schur_complement_local_range_partial = collect(1:bottom_vec_global_size)
        local_top_vector_range_partial = 1:top_vec_local_size
        local_top_vector_unique_entries_partial = local_top_vector_unique_entries
        local_top_vector_repeats_partial = local_top_vector_repeats
        global_bottom_vector_range_partial = owned_bottom_vector_entries
        local_bottom_vector_range_partial = 1:bottom_vec_local_size
        global_bottom_vector_entries_no_overlap_partial = owned_bottom_vector_entries_no_overlap
        local_bottom_vector_entries_no_overlap_partial =
            find_local_vector_inds(owned_bottom_vector_entries_no_overlap,
                                   owned_bottom_vector_entries)
        local_bottom_vector_repeats_partial = local_bottom_vector_repeats
        B_local_column_repeats_partial = local_bottom_vector_repeats
        C_local_row_repeats_partial = C_local_row_repeats
        D_local_column_repeats_partial = D_local_column_repeats
    else
        if synchronize_shared === nothing
            synchronize_shared = ()->MPI.Barrier(shared_comm)
        end

        function get_shared_partial_ranges(global_range, local_range)
            # Select a subset of global_range and local_range that will be handled
            # locally.
            if length(global_range) != length(local_range)
                error("expected global_range and local_range to have the same lengths. "
                      * "Got length(global_range)=$(length(global_range)), "
                      * "length(local_range)=$(length(local_range)).")
            end

            n = length(global_range)
            local_n_list = fill(n ÷ shared_nproc, shared_nproc)
            n_missing = n - sum(local_n_list)
            # Add extra points to processes so that local_n_list adds up to n.
            # n_missing will always be less than shared_nproc.
            for i ∈ 0:n_missing-1
                local_n_list[end-i] += 1
            end
            imin = sum(local_n_list[1:shared_rank]) + 1
            imax = imin + local_n_list[shared_rank+1] - 1

            this_range = imin:imax
            new_global_range = global_range[imin:imax]
            new_local_range = local_range[imin:imax]

            return new_global_range, new_local_range
        end
        function get_shared_partial_entries(global_entries, local_entries)
            # Select a subset of global_entries that will be handled locally.
            n = length(global_entries)
            local_n_list = fill(n ÷ shared_nproc, shared_nproc)
            n_missing = n - sum(local_n_list)
            # Add extra points to processes so that local_n_list adds up to n.
            # n_missing will always be less than shared_nproc.
            for i ∈ 0:n_missing-1
                local_n_list[end-i] += 1
            end
            imin = sum(local_n_list[1:shared_rank]) + 1
            imax = imin + local_n_list[shared_rank+1] - 1

            return global_entries[imin:imax], local_entries[imin:imax]
        end

        B_column_range_partial, _ =
            get_shared_partial_ranges(1:bottom_vec_global_size, 1:bottom_vec_global_size)
        B_global_column_range_partial, B_local_column_range_partial =
            get_shared_partial_ranges(B_global_column_range, B_local_column_range)
        C_global_row_range_partial, C_local_row_range_partial =
            get_shared_partial_ranges(C_global_row_range, C_local_row_range)
        D_global_column_range_partial, D_local_column_range_partial =
            get_shared_partial_ranges(D_global_column_range, D_local_column_range)
        _, schur_complement_local_range_partial =
            get_shared_partial_ranges(collect(1:bottom_vec_global_size),
                                      collect(1:bottom_vec_global_size))
        _, local_top_vector_range_partial =
            get_shared_partial_ranges(owned_top_vector_entries, 1:top_vec_local_size)
        _, local_top_vector_unique_entries_partial =
            get_shared_partial_ranges(unique_top_vector_entries,
                                      local_top_vector_unique_entries)
        global_bottom_vector_range_partial, local_bottom_vector_range_partial =
            get_shared_partial_ranges(owned_bottom_vector_entries,
                                      1:bottom_vec_local_size)
        global_bottom_vector_entries_no_overlap_partial,
        local_bottom_vector_entries_no_overlap_partial =
            get_shared_partial_entries(owned_bottom_vector_entries_no_overlap,
                                       find_local_vector_inds(owned_bottom_vector_entries_no_overlap,
                                                              owned_bottom_vector_entries))
        local_top_vector_repeats_partial =
            get_partial_repeated_inds(local_top_vector_repeats,
                                      local_top_vector_unique_entries_partial)
        local_bottom_vector_repeats_partial =
            get_partial_repeated_inds(local_bottom_vector_repeats,
                                      local_bottom_vector_range_partial)
        B_local_column_repeats_partial =
            get_partial_repeated_inds(local_bottom_vector_repeats,
                                      B_local_column_range_partial)
        C_local_row_repeats_partial = get_partial_repeated_inds(C_local_row_repeats,
                                                                C_local_row_range_partial)
        D_local_column_repeats_partial = get_partial_repeated_inds(D_local_column_repeats,
                                                                   D_local_column_range_partial)
    end
    C_row_counter = zeros(Int64, top_vec_local_size)

    # Allocate buffer arrays
    if sparse_Ainv_B && !use_sparse
        error("`use_sparse` must be true when `sparse_Ainv_B=true`.")
    end
    if isa(Ainv_dot_B_buffer, MPISchurComplementBlockAinvDotB) && !isa(C_buffer, MPISchurComplementBlockC)
        error("When an MPISchurComplementBlockAinvDotB is passed for Ainv_dot_B_buffer, "
              * "a MPISchurComplementBlockC must be passed for C_buffer.")
    end
    if !isa(Ainv_dot_B_buffer, MPISchurComplementBlockAinvDotB) && isa(C_buffer, MPISchurComplementBlockC)
        error("When an MPISchurComplementBlockC is passed for C_buffer, "
              * "a MPISchurComplementBlockAinvDotB must be passed for Ainv_dot_B_buffer.")
    end
    if isa(Ainv_dot_B_buffer, MPISchurComplementBlockAinvDotB) && separate_Ainv_B
        error("When an MPISchurComplementBlockAinvDotB is passed for Ainv_dot_B_buffer, "
              * "separate_Ainv_B must be false.")
    end
    if Ainv_dot_B_buffer === nothing
        if sparse_Ainv_B
            error("Ainv_dot_B_buffer is required when sparse_Ainv_B=true")
        end
        Ainv_dot_B = allocate_shared_float(top_vec_local_size, bottom_vec_global_size)
    else
        Ainv_dot_B = Ainv_dot_B_buffer
    end
    if separate_Ainv_B
        if !use_sparse
            error("It will always be more expensive to use `separate_Ainv_B` when "
                  * "`use_sparse=false`.")
        end
        Ainv_dot_B_local = nothing
        B_local = sparse(zeros(data_type, length(local_top_vector_unique_entries_partial),
                               bottom_vec_global_size))
    elseif isa(Ainv_dot_B_buffer, MPISchurComplementBlockAinvDotB)
        # Parallelisation of Ainv_dot_B*y is handled by the externally implemented
        # Ainv_dot_B_dot_y!(), so Ainv_dot_B_local is not needed.
        Ainv_dot_B_local = nothing
        B_local = nothing
    elseif sparse_Ainv_B
        # Store the chunk of Ainv_dot_B needed by this shared-memory process as a sparse
        # array.
        Ainv_dot_B_local = spzeros(data_type,
                                   length(local_top_vector_unique_entries_partial),
                                   bottom_vec_global_size)
        B_local = nothing
    else
        # Store the chunk of Ainv_dot_B needed by this shared-memory process as a contiguous
        # array.
        # Note that we need to transpose Ainv_dot_B_local for the slightly hacked
        # matrix-vector multiply implementation used in `ldiv!()` to ensure consistency of
        # results.
        Ainv_dot_B_local = Matrix{data_type}(undef, bottom_vec_global_size,
                                             length(local_top_vector_unique_entries_partial))
        B_local = nothing
    end
    Ainv_dot_u = allocate_shared_float(top_vec_local_size)
    # C_dot_Ainv_dot_u and C_dot_Ainv_dot_B are purely local buffers.
    C_dot_Ainv_dot_u = Vector{data_type}(undef, length(C_global_row_range_partial))
    C_dot_Ainv_dot_B_storage = nothing
    if sparse_Ainv_B
        if isa(Ainv_dot_B, MPISchurComplementBlockAinvDotB) && isa(C_buffer, MPISchurComplementBlockC)
            if !isa(schur_complement_buffer, FixedSparseCSC)
                error("Currently using MPISchurComplementBlockAinvDotB and "
                      * "MPISchurComplementBlockC is only supported when "
                      * "schur_complement_buffer is a FixedSparseCSC.")
            else
                # Need to collect contributions to C_dot_Ainv_dot_B from different processes,
                # then combine them into schur_complement. We do this by having a copy of
                # C_dot_Ainv_dot_B for each shared-memory process stored in shared memory,
                # then once each process calculates its contribution, the contributions from
                # different processes can be summed.
                if C_dot_Ainv_dot_B_buffer_ncopies === nothing
                    C_dot_Ainv_dot_B_buffer_ncopies = shared_nproc
                end
                schur_nnz = nnz(schur_complement_buffer)
                C_dot_Ainv_dot_B_storage =
                    allocate_shared_float(C_dot_Ainv_dot_B_buffer_ncopies, schur_nnz)
                if shared_rank == 0
                    C_dot_Ainv_dot_B_storage .= 0.0
                end
                C_dot_Ainv_dot_B = (colptr=schur_complement_buffer.colptr,
                                    rowval=schur_complement_buffer.rowval,
                                    storage=C_dot_Ainv_dot_B_storage)

                # B_column_range_partial is not otherwise used in this case (Ainv_dot_B is
                # a MPISchurComplementBlockAinvDotB and C_buffer is a
                # MPISchurComplementBlockC), but it is a UnitRange, so we can abuse it to
                # hold the range of flattened indices that this process should handle when
                # updating schur_complement_matrix with C_dot_Ainv_dot_B.
                n_flat_per_proc = (schur_nnz + shared_nproc - 1) ÷ shared_nproc
                B_column_range_partial = shared_rank*n_flat_per_proc+1:min((shared_rank+1)*n_flat_per_proc,schur_nnz)
            end
        elseif isa(schur_complement_buffer, FixedSparseCSC)
            C_dot_Ainv_dot_B =
                get_partial_FixedSparseCSC_buffer(C_global_row_range_partial,
                                                  schur_complement_buffer, data_type)
        else
            C_dot_Ainv_dot_B = spzeros(data_type, length(C_global_row_range_partial),
                                       bottom_vec_global_size)
        end
    else
        C_dot_Ainv_dot_B = Matrix{data_type}(undef, length(C_global_row_range_partial),
                                             bottom_vec_global_size)
    end
    if separate_Ainv_B
        Ainv_dot_B_dot_y = Vector{data_type}(undef, length(local_top_vector_unique_entries_partial))
    else
        Ainv_dot_B_dot_y = nothing
    end
    if schur_complement_buffer === nothing
        schur_complement = allocate_shared_float(bottom_vec_global_size, bottom_vec_global_size)
    else
        schur_complement = schur_complement_buffer
    end
    top_vec_buffer = allocate_shared_float(top_vec_local_size)
    if sparse_Ainv_B
        local_top_vec_buffer = Vector{data_type}(undef, length(local_top_vector_unique_entries_partial))
    else
        local_top_vec_buffer = nothing
    end
    bottom_vec_buffer = allocate_shared_float(bottom_vec_global_size)
    global_y = allocate_shared_float(bottom_vec_global_size)

    if C_buffer === nothing
        if use_sparse
            C_buffer = sparsecsr(Int64[], Int64[], data_type[],
                                 length(C_local_row_range_partial), top_vec_local_size)
            #C_buffer = transpose(sparse(Int64[], Int64[], data_type[], top_vec_local_size,
            #                            length(C_local_row_range_partial)))
        else
            C_buffer = zeros(data_type, length(C_local_row_range_partial),
                             top_vec_local_size)
        end
    end

    if isa(parallel_schur, Bool) && parallel_schur
        if schur_tile_size === nothing
            power_of_2 = floor(Int64, log2(bottom_vec_global_size / 2))
            schur_tile_size = min(128, 2^power_of_2)
        end
        schur_complement_factorization =
            mpi_dense_lu(schur_complement, schur_tile_size, comm, shared_comm,
                         distributed_comm, allocate_shared_float, allocate_shared_int;
                         synchronize_shared=synchronize_shared, skip_factorization=true,
                         check_lu=check_lu, timer=timer)
    elseif isa(parallel_schur, Bool)
        if shared_rank == 0 && distributed_rank == 0
            schur_complement_factorization =
                lu!(Matrix{data_type}(I, bottom_vec_global_size, bottom_vec_global_size);
                    check=check_lu)
        else
            schur_complement_factorization = nothing
        end
    else
        # Use a user-provided solver for Schur complement matrix solve.
        schur_complement_factorization = parallel_schur
        parallel_schur = true
    end

    if copy_input_to_dense_buffers
        B_dense_buffer = allocate_shared_float(top_vec_local_size, bottom_vec_local_size)
        C_dense_buffer = allocate_shared_float(bottom_vec_local_size, top_vec_local_size)
        D_dense_buffer = allocate_shared_float(bottom_vec_local_size, bottom_vec_local_size)
        if shared_rank == 0
            B_dense_buffer .= 0.0
            C_dense_buffer .= 0.0
            D_dense_buffer .= 0.0
        end
        synchronize_shared()
    else
        B_dense_buffer = nothing
        C_dense_buffer = nothing
        D_dense_buffer = nothing
    end

    sc_factorization = MPISchurComplement(A_factorization, Ainv_dot_B, Ainv_dot_B_local,
                                          B_local, B_column_range_partial,
                                          B_global_column_range,
                                          B_global_column_range_partial,
                                          B_local_column_range,
                                          B_local_column_range_partial,
                                          B_local_column_repeats,
                                          B_local_column_repeats_partial, B_dense_buffer,
                                          C_buffer, C_global_row_range_partial,
                                          C_local_row_range_partial,
                                          C_local_row_repeats_partial,
                                          C_row_counter, C_dense_buffer,
                                          D_global_column_range_partial,
                                          D_local_column_range_partial,
                                          D_local_column_repeats,
                                          D_local_column_repeats_partial, D_dense_buffer,
                                          schur_complement,
                                          schur_complement_factorization,
                                          schur_complement_local_range_partial,
                                          Ainv_dot_u, C_dot_Ainv_dot_B, C_dot_Ainv_dot_u,
                                          Ainv_dot_B_dot_y, top_vec_buffer,
                                          local_top_vec_buffer, top_vec_local_size,
                                          bottom_vec_buffer, bottom_vec_local_size,
                                          global_y, top_vec_global_size,
                                          bottom_vec_global_size,
                                          owned_top_vector_entries,
                                          local_top_vector_range_partial,
                                          local_top_vector_unique_entries_partial,
                                          local_top_vector_overlaps,
                                          local_top_vector_repeats,
                                          local_top_vector_repeats_partial,
                                          B_overlap_buffers_send, B_overlap_buffers_recv,
                                          overlap_ranks, owned_bottom_vector_entries,
                                          unique_bottom_vector_entries,
                                          global_bottom_vector_range_partial,
                                          global_bottom_vector_entries_no_overlap_partial,
                                          local_bottom_vector_range_partial,
                                          local_bottom_vector_entries_no_overlap_partial,
                                          local_bottom_vector_unique_entries,
                                          local_bottom_vector_repeats,
                                          local_bottom_vector_repeats_partial,
                                          comm, shared_comm, shared_rank,
                                          distributed_comm, distributed_rank,
                                          distributed_nproc, synchronize_shared,
                                          use_sparse, separate_Ainv_B, parallel_schur,
                                          check_lu, timer)

    if !skip_factorization
        update_schur_complement!(sc_factorization, missing, B, C, D)
    end

    return sc_factorization
end

function update_A_factorization!(sc::MPISchurComplement, A)
    timer = sc.timer
    A_factorization = sc.A_factorization

    # When `A===missing`, this was called from the `mpi_schur_complement()` constructor,
    # where we assume `A_factorization` was already initialized.
    if A !== missing
        @sc_timeit timer "lu(A)" begin
            lu!(A_factorization, A)
        end
    end
    return nothing
end

function update_Ainv_dot_B!(sc, B)
    timer = sc.timer
    A_factorization = sc.A_factorization
    Ainv_dot_B = sc.Ainv_dot_B
    local_top_vector_repeats = sc.local_top_vector_repeats
    local_bottom_vector_repeats = sc.local_bottom_vector_repeats
    B_column_range_partial = sc.B_column_range_partial
    B_local_column_repeats_partial = sc.B_local_column_repeats_partial
    B_local_column_range_partial = sc.B_local_column_range_partial
    B_global_column_range_partial = sc.B_global_column_range_partial
    overlap_ranks = sc.overlap_ranks
    local_top_vector_repeats_partial = sc.local_top_vector_repeats_partial
    local_top_vector_unique_entries_partial = sc.local_top_vector_unique_entries_partial
    local_top_vector_overlaps = sc.local_top_vector_overlaps
    B_overlap_buffers_send = sc.B_overlap_buffers_send
    B_overlap_buffers_recv = sc.B_overlap_buffers_recv
    separate_Ainv_B = sc.separate_Ainv_B
    shared_rank = sc.shared_rank
    distributed_nproc = sc.distributed_nproc
    distributed_comm = sc.distributed_comm
    synchronize_shared = sc.synchronize_shared

    @sc_timeit timer "Ainv_dot_B" begin
        # Use `Ainv_dot_B` as a local-rows/global-columns sized buffer to collect `B` into.
        # This is slightly inefficient, as there will be chunks that are all-zero that we do
        # not need to collect, but this way seems the simplest to implement, as we need the
        # not-locally-owned columns of B in all locally owned rows, to pass to `ldiv!()`
        # below.
        # When there are repeated entries in `B`, need to add them up into a single entry, and
        # then copy this entry into all the repeated positions. This converts the columns of
        # `B` into 'vectors' (with the same structure as `u`) that can be passed to
        # `A_factorization` to find `Ainv_dot_B`.
        if isa(Ainv_dot_B, MPISchurComplementBlockAinvDotB)
            # When using MPISchurComplementBlockAinvDotB, any zeroing of Ainv_dot_B is
            # taken care of within copy_B_submatrix!().
        elseif isa(Ainv_dot_B, AbstractSparseMatrix)
            Ainv_dot_B_colptr = Ainv_dot_B.colptr
            Ainv_dot_B_nzval = Ainv_dot_B.nzval
            Ainv_dot_B_first_i = Ainv_dot_B_colptr[first(B_column_range_partial)]
            Ainv_dot_B_last_i = Ainv_dot_B_colptr[last(B_column_range_partial)+1] - 1
            Ainv_dot_B_nzval[Ainv_dot_B_first_i:Ainv_dot_B_last_i] .= 0.0
        else
            Ainv_dot_B[:,B_column_range_partial] .= 0
        end
        if length(local_top_vector_repeats) > 0 || length(local_bottom_vector_repeats) > 0
            # Add up entries that are repeated on this subdomain.
            for j ∈ 1:size(B, 2), (to, from) ∈ eachcol(local_top_vector_repeats_partial)
                B[to,j] += B[from,j]
            end
            synchronize_shared()
            for (to, from) ∈ eachcol(B_local_column_repeats_partial)
                @views B[:,to] .+= B[:,from]
            end
            if isa(Ainv_dot_B, MPISchurComplementBlockAinvDotB)
                # B_local_column_repeats_partial will not necessarily correspond to the
                # indices of the locally-owned blocks in a MPISchurComplementBlockAinvDotB
                # (as they do to entries in B_local_column_range_partial), so need to
                # synchronize here. Currently expect that when
                # MPISchurComplementBlockAinvDotB is being used, there are no overlaps to
                # handle, so this block will not be entered - if it is used could consider
                # trying to adjust index ranges to avoid this synchronization?
                synchronize_shared()
            end
        else
            # When using MPISchurComplementBlockAinvDotB, any zeroing of Ainv_dot_B is
            # taken care of within copy_B_submatrix!(), so that there is no need for
            # synchronization here.
            synchronize_shared()
        end
        if isa(Ainv_dot_B, MPISchurComplementBlockAinvDotB)
            copy_B_submatrix!(Ainv_dot_B, B)
        elseif isa(Ainv_dot_B, AbstractSparseMatrix)
            update_sparse_matrix_select_columns!(Ainv_dot_B, B_global_column_range_partial,
                                                 B, B_local_column_range_partial)
            synchronize_shared()
        else
            for (j1, j2) ∈ zip(B_global_column_range_partial, B_local_column_range_partial), i ∈ 1:size(Ainv_dot_B, 1)
                Ainv_dot_B[i,j1] = B[i,j2]
            end
            synchronize_shared()
        end

        # Add up the rows of B that overlap between different subdomains (temporarily stored
        # in `Ainv_dot_B`).  Note only non-repeated points in the overlaps are communicated,
        # to reduce the amount of communication.
        if isa(Ainv_dot_B, MPISchurComplementBlockAinvDotB)
            # When using MPISchurComplementBlockAinvDotB, it is guaranteed that there are
            # no overlaps that need to be handled using distributed MPI, so can skip the
            # next step.
        else
            if length(overlap_ranks) > 0 && shared_rank == 0
                reqs = MPI.Request[]
                for (overlap_range, buffer_send, buffer_recv, overlap_rank) ∈
                        zip(local_top_vector_overlaps, B_overlap_buffers_send,
                            B_overlap_buffers_recv, overlap_ranks)
                    for j ∈ 1:size(Ainv_dot_B, 2), (i1, i2) ∈ enumerate(overlap_range)
                        buffer_send[i1,j] = Ainv_dot_B[i2,j]
                    end
                    # Iallreduce seems not to be included in the nice Julia API, so have to use
                    # lower level call here.
                    push!(reqs, MPI.Isend(buffer_send, distributed_comm; dest=overlap_rank))
                    push!(reqs, MPI.Irecv!(buffer_recv, distributed_comm; source=overlap_rank))
                end
                MPI.Waitall(reqs)
                for (overlap_range, buffer) ∈ zip(local_top_vector_overlaps, B_overlap_buffers_recv)
                    for j ∈ 1:size(Ainv_dot_B, 2), (i2, i1) ∈ enumerate(overlap_range)
                        Ainv_dot_B[i1,j] += buffer[i2,j]
                    end
                end
            end
            synchronize_shared()
            if length(local_top_vector_repeats) > 0
                # Now that overlaps have been comunicated, all contributions have been added the
                # 'to' places, so we can now copy back these periodic entries to the 'from'
                # places.
                for j ∈ 1:size(Ainv_dot_B, 2), (to, from) ∈ eachcol(local_top_vector_repeats_partial)
                    Ainv_dot_B[from,j] = Ainv_dot_B[to,j]
                end
                if !separate_Ainv_B
                    synchronize_shared()
                end
            end
        end

        # At this point `Ainv_dot_B` contains the dense array of `B`.
        if separate_Ainv_B
            if issparse(Ainv_dot_B)
                update_sparse_matrix_select_rows!(sc.B, Ainv_dot_B,
                                                  local_top_vector_unique_entries_partial)
            else
                sc_B = sc.B
                for j ∈ 1:size(Ainv_dot_B, 2), (i1, i2) ∈ enumerate(local_top_vector_unique_entries_partial)
                    sc_B[i1,j] = Ainv_dot_B[i2,j]
                end
            end
            synchronize_shared()
        end
        if isa(A_factorization, MPISchurComplementAFactorization) || isa(Ainv_dot_B, MPISchurComplementBlockAinvDotB)
            ldiv_Bmatrix!(A_factorization, Ainv_dot_B)
        else
            ldiv!(A_factorization, Ainv_dot_B)
        end
    end
    return nothing
end

function update_C!(sc, C)
    timer = sc.timer
    C_local_row_repeats_partial = sc.C_local_row_repeats_partial
    sc_C = sc.C

    @sc_timeit timer "C" begin
        if isa(sc_C, MPISchurComplementBlockC)
            copy_C_submatrix!(sc_C, C)
        else
            # A representation of C is stored where no rows are repeated, so need to add
            # up all contributions from repeated row indices into a single row (that will
            # then be included in the stored `sc.C`, i.e. the 'to' rows are included in
            # `sc.C_local_row_range_partial` while the 'from' rows are not).
            for j ∈ 1:size(C, 2), (to, from) ∈ eachcol(C_local_row_repeats_partial)
                C[to,j] += C[from,j]
            end
            # When using shared memory, only store the slice of C that this process needs.
            if issparse(sc_C)
                update_sparse_matrix!(sc_C, C, sc.C_local_row_range_partial,
                                      sc.C_row_counter)
            else
                # Make a copy because C_local_row_range_partial might not be a contiguous
                # range of indices, but performance will be better if `C` is a
                # contiguously-allocated array.
                for j ∈ 1:size(C, 2), (i1, i2) ∈ enumerate(sc.C_local_row_range_partial)
                    sc_C[i1,j] = C[i2,j]
                end
            end
        end
    end
    return nothing
end

function update_schur_complement_factorization!(sc, D)
    timer = sc.timer
    schur_complement = sc.schur_complement
    schur_complement_local_range_partial = sc.schur_complement_local_range_partial
    separate_Ainv_B = sc.separate_Ainv_B
    Ainv_dot_B = sc.Ainv_dot_B
    Ainv_dot_B_local = sc.Ainv_dot_B_local
    local_top_vector_unique_entries_partial = sc.local_top_vector_unique_entries_partial
    this_C = sc.C
    C_dot_Ainv_dot_B = sc.C_dot_Ainv_dot_B
    C_global_row_range_partial = sc.C_global_row_range_partial
    schur_complement = sc.schur_complement
    schur_complement_factorization = sc.schur_complement_factorization
    local_bottom_vector_repeats = sc.local_bottom_vector_repeats
    local_bottom_vector_repeats_partial = sc.local_bottom_vector_repeats_partial
    D_local_column_repeats = sc.D_local_column_repeats
    D_local_column_repeats_partial = sc.D_local_column_repeats_partial
    D_global_column_range_partial = sc.D_global_column_range_partial
    D_local_column_range_partial = sc.D_local_column_range_partial
    unique_bottom_vector_entries = sc.unique_bottom_vector_entries
    local_bottom_vector_unique_entries = sc.local_bottom_vector_unique_entries
    distributed_comm = sc.distributed_comm
    shared_rank = sc.shared_rank
    distributed_nproc = sc.distributed_nproc
    distributed_comm = sc.distributed_comm
    synchronize_shared = sc.synchronize_shared
    check_lu = sc.check_lu

    @sc_timeit timer "schur_complement" begin
        # Initialise `schur_complement` to zero, because when `this_C` does not include all rows,
        # the matrix multiplication below would not initialise all elements.
        if isa(Ainv_dot_B, MPISchurComplementBlockAinvDotB) && isa(this_C, MPISchurComplementBlockC)
            # Don't need to initialize schur_complement in this case.
        elseif issparse(schur_complement)
            schur_colptr = schur_complement.colptr
            schur_nzval = schur_complement.nzval
            for j ∈ schur_complement_local_range_partial
                for flat_i ∈ schur_colptr[j]:schur_colptr[j+1]-1
                    schur_nzval[j] = 0.0
                end
            end
        else
            nrows = size(schur_complement, 2)
            for j ∈ schur_complement_local_range_partial, i ∈ 1:nrows
                schur_complement[i,j] = 0.0
            end
        end
        synchronize_shared()

        # Read out the local entries of `Ainv_dot_B` here, rather than just after `Ainv_dot_B`
        # is calculated, in order to avoid adding another `synchronize_shared()` call.
        if !isa(Ainv_dot_B, MPISchurComplementBlockAinvDotB) && !separate_Ainv_B
            if isa(Ainv_dot_B_local, SparseMatrixCSC)
                # Convert Ainv_dot_B to SparseMatrixCSC in this call to resolve possible
                # type instability.
                update_sparse_matrix_select_rows!(Ainv_dot_B_local, Ainv_dot_B,
                                                  local_top_vector_unique_entries_partial)
            else
                # Note that we need to transpose Ainv_dot_B_local for the slightly
                # hacked matrix-vector multiply implementation used in `ldiv!()` to
                # ensure consistency of results.
                for i ∈ 1:size(Ainv_dot_B_local, 1), (j1, j2) ∈ enumerate(local_top_vector_unique_entries_partial)
                    Ainv_dot_B_local[i,j1] = Ainv_dot_B[j2,i]
                end
            end
        end

        # We store locally all columns in `Ainv_dot_B` (only local rows) and all rows of `C`
        # (only local columns). Therefore we can take the matrix product `Ainv_dot_B*C` with
        # the local chunks, then do a sum-reduce to get the final result. The
        # `schur_complement` buffer is full size on every rank.
        if isa(Ainv_dot_B, MPISchurComplementBlockAinvDotB) && isa(this_C, MPISchurComplementBlockC)
            mul_C_Ainv_dot_B!(C_dot_Ainv_dot_B, this_C, Ainv_dot_B)
            synchronize_shared()
        elseif isa(this_C, SparseMatrixCSR) && isa(Ainv_dot_B, AbstractSparseMatrixCSC)
            csr_mul!(C_dot_Ainv_dot_B, this_C, Ainv_dot_B, -1.0, 0.0)
        else
            mul!(C_dot_Ainv_dot_B, this_C, Ainv_dot_B, -1.0, 0.0)
        end
        if isa(Ainv_dot_B, MPISchurComplementBlockAinvDotB) && isa(this_C, MPISchurComplementBlockC)
            # Contributions from each process have now been calculated, and are available
            # in C_dot_Ainv_dot_B.storage. Need to add up all contributions into
            # schur_complement.
            # B_column_range_partial is abused to store the index range that we need for
            # this update, as it is not used otherwise.
            flat_range = sc.B_column_range_partial
            if !isempty(flat_range)
                @views sum!(schur_complement.nzval[flat_range]', sc.C_dot_Ainv_dot_B.storage[:,flat_range])
            end
        elseif issparse(C_dot_Ainv_dot_B)
            C_dot_Ainv_dot_B_colptr = C_dot_Ainv_dot_B.colptr
            C_dot_Ainv_dot_B_rowval = C_dot_Ainv_dot_B.rowval
            C_dot_Ainv_dot_B_nzval = C_dot_Ainv_dot_B.nzval
            for j ∈ 1:size(schur_complement, 2)
                for flat_i ∈ C_dot_Ainv_dot_B_colptr[j]:C_dot_Ainv_dot_B_colptr[j+1]-1
                    i = C_dot_Ainv_dot_B_rowval[flat_i]
                    global_i = C_global_row_range_partial[i]
                    schur_complement[global_i,j] = C_dot_Ainv_dot_B_nzval[flat_i]
                end
            end
        else
            for j ∈ 1:size(schur_complement, 2), (i2, i1) ∈ enumerate(C_global_row_range_partial)
                schur_complement[i1,j] = C_dot_Ainv_dot_B[i2,j]
            end
        end
        synchronize_shared()
        # Only get the local rows for D, so just add these to the local rows of
        # `schur_complement`.
        # As `schur_Complement` does not have any repeated entries, need to add up any locally
        # repeated entries of D (columns then rows) so that we can then select the 'assembled'
        # version of `D` to add to `schur_complement`. Any entries that are repeated on
        # different subdomains will be taken care of when the local contributions to
        # `schur_complement` are added together in the `MPI.Reduce!()` below, as there may be
        # non-zero contributions to some entries from multiple subdomains.
        if length(local_bottom_vector_repeats) > 0
            for j ∈ 1:size(D, 2), (to, from) ∈ eachcol(local_bottom_vector_repeats_partial)
                D[to,j] += D[from,j]
            end
            synchronize_shared()
        end
        if length(D_local_column_repeats) > 0
            for (to, from) ∈ eachcol(D_local_column_repeats_partial)
                @views D[:,to] .+= D[:,from]
            end
        end
        if issparse(D) && issparse(schur_complement)
            D_full = parent(D)
            full_rowinds, full_colinds = D.indices
            D_colptr = D_full.colptr
            D_rowval = D_full.rowval
            D_nzval = D_full.nzval
            sc_colptr = schur_complement.colptr
            sc_rowval = schur_complement.rowval
            sc_nzval = schur_complement.nzval
            nrow = length(unique_bottom_vector_entries)
            for (j1, j2) ∈ zip(D_global_column_range_partial, D_local_column_range_partial)
                full_j2 = full_colinds[j2]
                first_i = sc_colptr[j1]
                last_i = sc_colptr[j1+1] - 1
                if last_i < first_i
                    continue
                end
                # Assume D and schur_complement have same pattern of non-zeros, so no need
                # to use searchsortedlast() to find the first flat_i that will be within
                # the non-zeros of D.
                flat_i = first_i

                full_first_i = D_colptr[full_j2]
                full_last_i = D_colptr[full_j2+1]-1
                if full_last_i < full_first_i
                    continue
                end

                first_row = sc_rowval[first_i]
                row_counter = max(searchsortedlast(unique_bottom_vector_entries, first_row) - 1, 1)
                for full_flat_i ∈ full_first_i:full_last_i
                    full_row = D_rowval[full_flat_i]
                    while row_counter ≤ nrow && full_rowinds[local_bottom_vector_unique_entries[row_counter]] < full_row
                        row_counter += 1
                    end
                    if row_counter > nrow
                        break
                    end
                    sc_row = unique_bottom_vector_entries[row_counter]
                    while flat_i ≤ last_i && sc_rowval[flat_i] < sc_row
                        flat_i += 1
                    end
                    if flat_i > last_i
                        continue
                    end
                    if full_rowinds[sc_rowval[flat_i]] == full_row
                        sc_nzval[flat_i] += D_nzval[full_flat_i]
                    end
                end
            end
        else
            for (j1, j2) ∈ zip(D_global_column_range_partial, D_local_column_range_partial), (i1, i2) ∈ zip(unique_bottom_vector_entries, local_bottom_vector_unique_entries)
                schur_complement[i1,j1] += D[i2,j2]
            end
        end
        synchronize_shared()
        if shared_rank == 0 && distributed_nproc > 1
            MPI.Reduce!(schur_complement, +, distributed_comm; root=0)
        end

        if isa(schur_complement_factorization, LU)
            # `schur_complement` has been gathered/assembled onto the global rank-0
            # process, and is now LU-factorized in serial.
            # Unless the original matrices were all block-diagonal in some consistent
            # way (in which case the solve could probably be done more efficiently by
            # splitting the full matrix into the disconnected pieces),
            # `schur_complement` will generally be a dense matrix, so not worth having
            # an option for a sparse LU factorization here. Possibly this LU
            # factorization (and the corresponding `ldiv!()` using
            # `schur_complement_factorization`) could be parallelised with shared
            # memory and/or distributed MPI, but we expect this step not to be a
            # bottleneck, so it is done in serial (at least for now).
            factors = schur_complement_factorization.factors
            factors .= schur_complement
            ipiv = schur_complement_factorization.ipiv
            LAPACK.getrf!(factors, ipiv; check=check_lu)
        elseif !isa(schur_complement_factorization, Nothing)
            synchronize_shared()
            lu!(schur_complement_factorization, schur_complement)
        end
    end

    return nothing
end

"""
    update_schur_complement!(sc::MPISchurComplement, A, B::AbstractMatrix,
                             C::AbstractMatrix, D::AbstractMatrix)

Update the matrix which is being solved by `sc`.

`A` will be passed to `lu!(sc.A_factorization, A)`, so should be as required by the LU
implementation being used for `sc.A_factorization`.

`B`, `C`, and `D` should be the same shapes, and represent the same global index ranges,
as the inputs to `mpi_schur_complement()` used to construct `sc`. `B`, `C`, and `D` may be
modified.
"""
function update_schur_complement!(sc::MPISchurComplement, A, B::AbstractMatrix,
                                  C::AbstractMatrix, D::AbstractMatrix)
    timer = sc.timer
    @sc_timeit timer "update_schur_complement" begin
        @boundscheck isa(sc.Ainv_dot_B, MPISchurComplementBlockAinvDotB) || size(sc.Ainv_dot_B, 1) == size(B, 1) || error(BoundsError, " Number of rows in B does not match number of rows in original Ainv_dot_B")
        @boundscheck length(sc.B_local_column_range) + size(sc.B_local_column_repeats, 2) == size(B, 2) || error(BoundsError, " Number of columns in B does not match number of columns in original B")
        # Don't check size(C, 1) because we don't store the full row range. There will be an
        # out of bounds error from indexing by sc.C_local_row_range_partial if C is too small.
        @boundscheck sc.top_vec_local_size == size(C, 2) || error(BoundsError, " Number of columns in C does not match original C")
        @boundscheck length(sc.owned_bottom_vector_entries) == size(D, 1) || error(BoundsError, " Number of rows in D does not match number of locally owned bottom_vector_entries")
        @boundscheck length(sc.D_local_column_range_partial) == 0 || maximum(sc.D_local_column_range_partial) ≤ size(D, 2) || error(BoundsError, " Number of columns in D is smaller than the largest index in D_local_column_range")

        B_dense_buffer = sc.B_dense_buffer
        C_dense_buffer = sc.C_dense_buffer
        D_dense_buffer = sc.D_dense_buffer
        if B_dense_buffer !== nothing && C_dense_buffer !== nothing && D_dense_buffer !== nothing
            local_top_vector_range_partial = sc.local_top_vector_range_partial
            local_bottom_vector_range_partial = sc.local_bottom_vector_range_partial
            update_from_sparse_matrix_select_columns!(B_dense_buffer,
                                                      local_bottom_vector_range_partial,
                                                      B,
                                                      local_bottom_vector_range_partial)
            update_from_sparse_matrix_select_columns!(C_dense_buffer,
                                                      local_top_vector_range_partial, C,
                                                      local_top_vector_range_partial)
            update_from_sparse_matrix_select_columns!(D_dense_buffer,
                                                      local_bottom_vector_range_partial,
                                                      D,
                                                      local_bottom_vector_range_partial)
            this_B = B_dense_buffer
            this_C = C_dense_buffer
            this_D = D_dense_buffer
        else
            this_B = B
            this_C = C
            this_D = D
        end

        update_A_factorization!(sc, A)
        if B_dense_buffer !== nothing && C_dense_buffer !== nothing && D_dense_buffer !== nothing
            sc.synchronize_shared()
        end
        update_Ainv_dot_B!(sc, this_B)
        update_C!(sc, this_C)
        update_schur_complement_factorization!(sc, this_D)
    end

    return nothing
end

"""
    ldiv!(x::AbstractVector, y::AbstractVector, sc::MPISchurComplement,
          u::AbstractVector, v::AbstractVector)

Solve the 2x2 block-structured matrix system
```math
\\left(\\begin{array}{cc}
A & B\\\\
C & D
\\end{array}\\right)
\\cdot
\\left(\\begin{array}{c}
x\\\\
y
\\end{array}\\right)
=
\\left(\\begin{array}{c}
u\\\\
v
\\end{array}\\right)
```
for \$x\$ and \$v\$.

Only the locally owned parts of `u` and `v` should be passed, and the locally owned parts
of the solution will be written into `x` and `y`. This means that `x` and `u` should
correspond to the global indices in `sc.owned_top_vector_entries` and `y` and `v` should
correspond to the global indices in `sc.owned_bottom_vector_entries`. When shared memory
is used, `x` and `y` must be shared-memory arrays (identical on every process in
`sc.shared_comm`). `u` and `v` should also be shared memory arrays (although non-shared
identical copies would also work).
"""
function ldiv!(x::AbstractVector, y::AbstractVector, sc::MPISchurComplement,
               u::AbstractVector, v::AbstractVector)
    @boundscheck size(sc.top_vec_buffer) == size(u) || error(BoundsError, " Size of u does not match size of top_vec_buffer")
    @boundscheck size(sc.top_vec_buffer) == size(x) || error(BoundsError, " Size of x does not match size of top_vec_buffer")
    @boundscheck (length(sc.owned_bottom_vector_entries),) == size(v) || error(BoundsError, " Size of v does not match size of bottom_vector_buffer")
    @boundscheck (length(sc.owned_bottom_vector_entries),) == size(y) || error(BoundsError, " Size of y does not match size of bottom_vector_buffer")

    timer = sc.timer
    @sc_timeit timer "ldiv!" begin
        distributed_comm = sc.distributed_comm
        distributed_rank = sc.distributed_rank
        distributed_nproc = sc.distributed_nproc
        parallel_schur = sc.parallel_schur
        A_factorization = sc.A_factorization
        Ainv_dot_B = sc.Ainv_dot_B
        Ainv_dot_B_local = sc.Ainv_dot_B_local
        schur_complement_factorization = sc.schur_complement_factorization
        Ainv_dot_u = sc.Ainv_dot_u
        C_dot_Ainv_dot_u = sc.C_dot_Ainv_dot_u
        Ainv_dot_B_dot_y = sc.Ainv_dot_B_dot_y
        top_vec_buffer = sc.top_vec_buffer
        local_top_vector_range_partial = sc.local_top_vector_range_partial
        local_top_vector_unique_entries_partial = sc.local_top_vector_unique_entries_partial
        local_top_vector_repeats = sc.local_top_vector_repeats
        local_top_vector_repeats_partial = sc.local_top_vector_repeats_partial
        bottom_vec_buffer = sc.bottom_vec_buffer
        global_bottom_vector_range_partial = sc.global_bottom_vector_range_partial
        global_bottom_vector_entries_no_overlap_partial = sc.global_bottom_vector_entries_no_overlap_partial
        local_bottom_vector_range_partial = sc.local_bottom_vector_range_partial
        local_bottom_vector_entries_no_overlap_partial = sc.local_bottom_vector_entries_no_overlap_partial
        schur_complement_local_range_partial = sc.schur_complement_local_range_partial
        shared_rank = sc.shared_rank
        distributed_nproc = sc.distributed_nproc
        synchronize_shared = sc.synchronize_shared

        @sc_timeit timer "Ainv.u" begin
            ldiv!(Ainv_dot_u, A_factorization, u)
        end

        @sc_timeit timer "v-C.Ainv.u" begin
            # Initialise to zero, because when C does not include all rows, the matrix
            # multiplication below would not initialise all elements.
            bottom_vec_buffer[schur_complement_local_range_partial] .= 0.0
            synchronize_shared()
            if isa(sc.C, MPISchurComplementBlockC)
                mul_C_dot_Ainv_dot_u!(bottom_vec_buffer, sc.C, Ainv_dot_u)
            else
                # Need all rows of C, but only the local columns - this is all that is
                # stored in sc.C.
                mul!(C_dot_Ainv_dot_u, sc.C, Ainv_dot_u, -1.0, 0.0)
                for (i2, i1) ∈ enumerate(sc.C_global_row_range_partial)
                    bottom_vec_buffer[i1] = C_dot_Ainv_dot_u[i2]
                end
            end
            synchronize_shared()

            # Only have the local entries of v, so add those to the local entries in
            # bottom_vec_buffer before reducing.
            # Need to avoid double counting of any overlapping entries in `v`.
            for (i1, i2) ∈ zip(global_bottom_vector_entries_no_overlap_partial, local_bottom_vector_entries_no_overlap_partial)
                bottom_vec_buffer[i1] += v[i2]
            end
            synchronize_shared()
        end

        @sc_timeit timer "global_y" begin
            # `global_y` is solved in serial on the global rank-0 process, and then communicated
            # back to all other processes.
            global_y = sc.global_y
            if sc.shared_rank == 0 && distributed_nproc > 1
                MPI.Reduce!(bottom_vec_buffer, +, distributed_comm; root=0)
            end

            if parallel_schur
                if distributed_nproc > 1
                    synchronize_shared()
                end
                ldiv!(global_y, schur_complement_factorization, bottom_vec_buffer)
                if distributed_nproc > 1
                    synchronize_shared()
                end
            else
                if shared_rank == 0 && distributed_rank == 0
                    ldiv!(global_y, schur_complement_factorization, bottom_vec_buffer)
                end
            end

            if sc.shared_rank == 0 && distributed_nproc > 1
                MPI.Bcast!(global_y, distributed_comm; root=0)
            end
            synchronize_shared()
        end

        @sc_timeit timer "Ainv.u-Ainv.B.y" begin
            # Need all columns of Ainv_dot_B_local, but only the local rows.
            if isa(Ainv_dot_B, MPISchurComplementBlockAinvDotB)
                Ainv_dot_B_dot_y!(top_vec_buffer, Ainv_dot_B, global_y)
            elseif sc.separate_Ainv_B
                # B_local is a sparse matrix, so this might sometimes be numerically cheaper than
                # multiplying by a dense, precomputed Ainv_dot_B_local`.
                mul!(Ainv_dot_B_dot_y, sc.B, global_y)
                top_vec_buffer[local_top_vector_unique_entries_partial] .= Ainv_dot_B_dot_y

                # Fill in any repeated entries in `top_vec_buffer`. 'to' and 'from' are kinda
                # back-to-front here because most of the time `vector_repeats` (and similar) are
                # used to gather the repeats from the 'from' entries into the 'to' entries, but
                # here we scatter them back in the other direction.
                if length(local_top_vector_repeats) > 0
                    for (to, from) ∈ eachcol(local_top_vector_repeats_partial)
                        top_vec_buffer[from] = top_vec_buffer[to]
                    end
                end

                synchronize_shared()

                ldiv!(A_factorization, top_vec_buffer)
            else
                if isa(Ainv_dot_B_local, SparseMatrixCSC)
                    local_top_vec_buffer = sc.local_top_vec_buffer
                    @views mul!(local_top_vec_buffer, Ainv_dot_B_local, global_y)
                    for (i2, i1) ∈ enumerate(local_top_vector_unique_entries_partial)
                        top_vec_buffer[i1] = local_top_vec_buffer[i2]
                    end
                else
                    # This commented-out implementation should probably be the most
                    # performant, but may result in inconsistent floating-point errors in
                    # results that should be identical (i.e. identical rows and RHS, but
                    # the rows are in a different place in the matrix). This would mean
                    # that downstream code might have to communicate to ensure exact
                    # consistency of the results.
                    #mul!(Ainv_dot_B_dot_y, Ainv_dot_B_local, global_y)
                    #for (i2, i1) ∈ enumerate(local_top_vector_unique_entries_partial)
                    #    top_vec_buffer[i1] = Ainv_dot_B_dot_y
                    #end
                    # The following implementation might be slightly less performant
                    # (although on a quick check in serial the difference is negligible -
                    # note that we have transposed Ainv_dot_B_local for this version for
                    # efficiency, so that the slice Ainv_dot_B_local[:,i] that we need is
                    # contiguous in memory), but should produce exactly consistent results
                    # for identical row/RHS inputs. The results then do not need to be
                    # communicated, which should more than compensate for any loss in
                    # performance of this step.
                    for (count, i) ∈ enumerate(local_top_vector_unique_entries_partial)
                        top_vec_buffer[i] = @views dot(Ainv_dot_B_local[:,count], global_y)
                    end
                end

                # Fill in any repeated entries in `top_vec_buffer`. 'to' and 'from' are kinda
                # back-to-front here because most of the time `vector_repeats` (and similar) are
                # used to gather the repeats from the 'from' entries into the 'to' entries, but
                # here we scatter them back in the other direction.
                if length(local_top_vector_repeats) > 0
                    for (to, from) ∈ eachcol(local_top_vector_repeats_partial)
                        top_vec_buffer[from] = top_vec_buffer[to]
                    end
                end
            end
            # Could possibly remove this synchronization when using
            # MPISchurComplementBlockAinvDotB for Ainv_dot_B, if
            # `local_top_vector_range_partial` was compatible with the parallelisation
            # used in Ainv_dot_B_dot_y!(top_vec_buffer, Ainv_dot_B, global_y)? On the
            # other hand, local_top_vector_range_partial is a UnitRange, while the
            # parallelisation in Ainv_dot_B_dot_y!() will usually imply some more
            # complicated indexing by a Vector{Int64}, so it is not clear that doing this
            # operation with that parallelisation would be more efficient, even though it
            # would remove a synchronization.
            synchronize_shared()
            for i ∈ local_top_vector_range_partial
                x[i] = Ainv_dot_u[i] - top_vec_buffer[i]
            end

            for (i1, i2) ∈ zip(local_bottom_vector_range_partial, global_bottom_vector_range_partial)
                y[i1] = global_y[i2]
            end
            synchronize_shared()
        end
    end

    return nothing
end
# Due to the use of intermediate buffer arrays, there is no chance of aliasing errors when
# returning the result in the input vectors, so just forward to the 5-argument function.
function ldiv!(sc::MPISchurComplement, u::AbstractVector, v::AbstractVector)
    return ldiv!(u, v, sc, u, v)
end

# Import FakeMPILUs to make it available for prototyping/testing in other packages.
# Not exported as part of public interface because it shouldn't be used in 'production'.
include("FakeMPILUs.jl")

end # module MPISchurComplements
