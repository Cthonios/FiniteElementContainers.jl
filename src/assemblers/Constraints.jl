# util for sparse matrices on gpus
# TODO make non-allocating through some mapreduce functionality
# COV_EXCL_START
KA.@kernel function __sptrace_kernel!(diagA, colptr, rowvals, nz)
    J = KA.@index(Global)
    col_start = colptr[J]
    col_end   = colptr[J + 1] - 1
    for k in col_start:col_end
        if rowvals[k] == J
            diagA[J] = nz[k]
        end
    end
end
# COV_EXCL_STOP

function __sptrace(A)
    backend = KA.get_backend(A)
    diagA = KA.zeros(backend, eltype(A), size(A, 1))
    kernel! = __sptrace_kernel!(backend)
    kernel!(diagA, A.colPtr, A.rowVal, A.nzVal, ndrange = size(A, 2))
    return sum(diagA)
end

# TODO this only work on CPU right now
function _adjust_matrix_entries_for_constraints!(
    A::SparseMatrixCSC, constraint_storage, ::KA.CPU;
    penalty_scale = 1.e6
)
    # first ensure things are the right size
    @assert size(A, 1) == size(A, 2)
    @assert length(constraint_storage) == size(A, 2)
  
    # hacky for now
    # need a penalty otherwise we get into trouble with
    # iterative linear solvers even for a simple poisson problem
    # TODO perhaps this should be optional somehow
    penalty = penalty_scale * tr(A) / size(A, 2)
  
    # now modify A => (I - G) * A + G
    nz = nonzeros(A)
    rowval = rowvals(A)
    for j in 1:size(A, 2)
        col_start = A.colptr[j]
        col_end   = A.colptr[j + 1] - 1
        for k in col_start:col_end
            # for (I - G) * A term
            nz[k] = (1. - constraint_storage[j]) * nz[k]

            # for + G term
            if rowval[k] == j
                @inbounds nz[k] = nz[k] + penalty * constraint_storage[j]
            end
        end
    end
  
    return nothing
end
  
# COV_EXCL_START
KA.@kernel function _adjust_matrix_entries_for_constraints_kernel!(
    colptr, rowvals, nz, size_A_2,
    constraint_storage, trA, penalty_scale
)
    J = KA.@index(Global)
    penalty = penalty_scale * trA / size_A_2
  
    # now modify A => (I - G) * A + G
  
    col_start = colptr[J]
    col_end   = colptr[J + 1] - 1
    for k in col_start:col_end
        # for (I - G) * A term
        nz[k] = (1. - constraint_storage[J]) * nz[k]

        # for + G term
        if rowvals[k] == J
            @inbounds nz[k] = nz[k] + penalty * constraint_storage[J]
        end
    end
end
# COV_EXCL_STOP
  
function _adjust_matrix_entries_for_constraints!(
    A, constraint_storage, backend::KA.Backend;
    penalty_scale = 1.e6
)
    # first ensure things are the right size
    @assert size(A, 1) == size(A, 2)
    @assert length(constraint_storage) == size(A, 2)
  
    # get trA ahead of time to save some allocations at kernel level
    trA = __sptrace(A)

    kernel! = _adjust_matrix_entries_for_constraints_kernel!(backend)
    kernel!(A.colPtr, A.rowVal, A.nzVal, size(A, 2), constraint_storage, trA, penalty_scale, ndrange = size(A, 2))
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
