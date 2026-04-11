_colval(A::GPUArrays.AbstractGPUSparseMatrixCSR) = A.colVal
_colval(A::SparseMatrixCSR) = A.colval
_rowptr(A::GPUArrays.AbstractGPUSparseMatrixCSR) = A.rowPtr
_rowptr(A::SparseMatrixCSR) = A.rowptr

function __sptrace(backend::KA.Backend, A::GPUArrays.AbstractGPUSparseMatrixCSC)
    diagA = KA.zeros(backend, eltype(A), size(A, 2))
    colptr = SparseArrays.getcolptr(A)
    rowvals = SparseArrays.rowvals(A)
    nz = SparseArrays.nonzeros(A)
    fec_foraxes(A, 2) do j
        col_start = colptr[j]
        col_end   = colptr[j + 1] - 1
        for k in col_start:col_end
            if rowvals[k] == j
                diagA[j] = nz[k]
            end
        end
    end
    return reduce(+, diagA)
end

function __sptrace(backend::KA.Backend, A::GPUArrays.AbstractGPUSparseMatrixCSR)
    diagA = KA.zeros(backend, eltype(A), size(A, 1))
    rowptr = A.rowPtr
    colvals = A.colVal
    nz = SparseArrays.nonzeros(A)

    fec_foraxes(A, 1) do i
        row_start = rowptr[i]
        row_end   = rowptr[i + 1] - 1
        for k in row_start:row_end
            if colvals[k] == i
                diagA[i] = nz[k]
            end
        end
    end
    return reduce(+, diagA)
end

function __sptrace(::KA.Backend, A::AbstractSparseMatrix)
    return tr(A)
end

function _sptrace(A::SparseMatrixCSR)
    return tr(A)
end

function _sptrace(A::AbstractSparseMatrix)
    return __sptrace(KA.get_backend(A), A)
end

function _adjust_matrix_entries_for_constraints!(
    A::T,
    constraint_storage;
    penalty_scale = 1.e6
) where T <: Union{
    <:GPUArrays.AbstractGPUSparseMatrixCSC,
    <:SparseArrays.AbstractSparseMatrixCSC
}
    # first ensure things are the right size
    @assert size(A, 1) == size(A, 2)
    @assert length(constraint_storage) == size(A, 2)
  
    # hacky for now
    # need a penalty otherwise we get into trouble with
    # iterative linear solvers even for a simple poisson problem
    # TODO perhaps this should be optional somehow
    penalty = penalty_scale * _sptrace(A) / size(A, 2)
    
    nz = SparseArrays.nonzeros(A)
    colptr = SparseArrays.getcolptr(A)
    rowvals = SparseArrays.rowvals(A)
    fec_foraxes(A, 2) do j
        col_start = colptr[j]
        col_end = colptr[j + 1] - 1
        for k in col_start:col_end
            # for (I - G) * A term
            nz[k] = (1. - constraint_storage[j]) * nz[k]

            # for + G term
            if rowvals[k] == j
                @inbounds nz[k] = nz[k] + penalty * constraint_storage[j]
            end
        end
    end
    return nothing
end

function _adjust_matrix_entries_for_constraints!(
    A::T,
    constraint_storage;
    penalty_scale = 1.e6
) where T <: Union{<:GPUArrays.AbstractGPUSparseMatrixCSR}
    @assert size(A, 1) == size(A, 2)
    @assert length(constraint_storage) == size(A, 1)

    penalty = penalty_scale * _sptrace(A) / size(A, 1)
    nz      = SparseArrays.nonzeros(A)
    rowptr = _rowptr(A)
    colvals = _colval(A)

    fec_foraxes(A, 1) do i
        row_start = rowptr[i]
        row_end   = rowptr[i + 1] - 1
        for k in row_start:row_end
            # for (I - G) * A term
            nz[k] = (1.0 - constraint_storage[i]) * nz[k]
            # for + G term
            if colvals[k] == i
                @inbounds nz[k] = nz[k] + penalty * constraint_storage[i]
            end
        end
    end

    return nothing
end

# Needed so we don't have a type piracy from defining
# KA.get_backend(::SparseMatricesCSR)
function _adjust_matrix_entries_for_constraints!(
    A::SparseMatrixCSR,
    constraint_storage;
    penalty_scale = 1.e6
)
    @assert size(A, 1) == size(A, 2)
    @assert length(constraint_storage) == size(A, 1)

    penalty = penalty_scale * _sptrace(A) / size(A, 1)
    nz      = SparseArrays.nonzeros(A)
    rowptr = _rowptr(A)
    colvals = _colval(A)

    for i in axes(A, 1)
        row_start = rowptr[i]
        row_end   = rowptr[i + 1] - 1
        for k in row_start:row_end
            # for (I - G) * A term
            nz[k] = (1.0 - constraint_storage[i]) * nz[k]
            # for + G term
            if colvals[k] == i
                @inbounds nz[k] = nz[k] + penalty * constraint_storage[i]
            end
        end
    end

    return nothing
end

function _adjust_matrix_action_entries_for_constraints!(
    Av, constraint_storage, v
)
    @assert length(Av) == length(constraint_storage)
    @assert length(v) == length(constraint_storage)   
    fec_foreach(constraint_storage) do n
        @inbounds Av[n] = (1. - constraint_storage[n]) * Av[n] + constraint_storage[n] * v[n]
    end
    return nothing
end

function _adjust_vector_entries_for_constraints!(b, constraint_storage)
    @assert length(b) == length(constraint_storage)
    fec_foreach(b) do n
        b[n] = (1. - constraint_storage[n]) * b[n]
    end
    return nothing
end
