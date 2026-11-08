module GPUArraysExt

import FiniteElementContainers as FEC
import GPUArrays
import KernelAbstractions as KA
import SparseArrays

function FEC._adjust_matrix_entries_for_constraints!(
    A::T,
    constraint_storage;
    penalty_scale = 1.e6
) where T <: Union{<:GPUArrays.AbstractGPUSparseMatrixCSR}
    @assert size(A, 1) == size(A, 2)
    @assert length(constraint_storage) == size(A, 1)

    penalty = penalty_scale * FEC._sptrace(A) / size(A, 1)
    nz      = SparseArrays.nonzeros(A)
    rowptr = FEC._rowptr(A)
    colvals = FEC._colval(A)

    FEC.fec_foraxes(A, 1) do i
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

FEC._colval(A::GPUArrays.AbstractGPUSparseMatrixCSR) = A.colVal
FEC._rowptr(A::GPUArrays.AbstractGPUSparseMatrixCSR) = A.rowPtr

function FEC.__sptrace(backend::KA.Backend, A::GPUArrays.AbstractGPUSparseMatrixCSC)
    diagA = KA.zeros(backend, eltype(A), size(A, 2))
    colptr = SparseArrays.getcolptr(A)
    rowvals = SparseArrays.rowvals(A)
    nz = SparseArrays.nonzeros(A)
    FEC.fec_foraxes(A, 2) do j
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

function FEC.__sptrace(backend::KA.Backend, A::GPUArrays.AbstractGPUSparseMatrixCSR)
    diagA = KA.zeros(backend, eltype(A), size(A, 1))
    rowptr = A.rowPtr
    colvals = A.colVal
    nz = SparseArrays.nonzeros(A)

    FEC.fec_foraxes(A, 1) do i
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

end # module
