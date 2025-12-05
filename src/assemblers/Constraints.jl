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
    A, constraint_storage, trA;
    penalty_scale = 1.e6
)
    J = KA.@index(Global)
  
    penalty = penalty_scale * trA / size(A, 2)
  
    # now modify A => (I - G) * A + G
    nz = nonzeros(A)
    rowval = rowvals(A)
  
    col_start = A.colptr[J]
    col_end   = A.colptr[J + 1] - 1
    for k in col_start:col_end
        # for (I - G) * A term
        nz[k] = (1. - constraint_storage[J]) * nz[k]

        # for + G term
        if rowval[k] == J
            @inbounds nz[k] = nz[k] + penalty * constraint_storage[J]
        end
    end
end
# COV_EXCL_STOP
  
function _adjust_matrix_entries_for_constraints(
    A, constraint_storage, backend::KA.Backend
)
    # first ensure things are the right size
    @assert size(A, 1) == size(A, 2)
    @assert length(constraint_storage) == size(A, 2)
  
    # get trA ahead of time to save some allocations at kernel level
    trA = tr(A)
    
    kernel! = _adjust_matrix_entries_for_constraints_kernel!(backend)
    kernel!(A, constraint_storage, trA, ndrange = size(A, 2))
    return nothing
end

function _adjust_matrix_action_entries_for_constraints!(
    Av, constraint_storage, v, ::KA.CPU
    # TODO do we need a penalty scale here as well?
  )
    @assert length(Av) == length(constraint_storage)
    @assert length(v) == length(constraint_storage)
    # modify Av => (I - G) * Av + Gv
    # TODO is this the right thing to do? I think so...
    for i in 1:length(constraint_storage)
        @inbounds Av[i] = (1. - constraint_storage[i]) * Av[i] + constraint_storage[i] * v[i]
    end
    return nothing
end
  
# COV_EXCL_START
KA.@kernel function _adjust_matrix_action_entries_for_constraints_kernel!(
    Av, constraint_storage, v
)
    I = KA.@index(Global)
    # modify Av => (I - G) * Av + Gv
    @inbounds Av[I] = (1. - constraint_storage[I]) * Av[I] + constraint_storage[I] * v[I]
end
# COV_EXCL_STOP
  
function _adjust_matrix_action_entries_for_constraints!(
    Av, constraint_storage, v, backend::KA.Backend
)
    @assert length(Av) == length(constraint_storage)
    @assert length(v) == length(constraint_storage)
    kernel! = _adjust_matrix_action_entries_for_constraints_kernel!(backend)
    kernel!(Av, constraint_storage, v, ndrange = length(Av))
    return nothing
end
  
function _adjust_vector_entries_for_constraints!(b, constraint_storage, ::KA.CPU)
    @assert length(b) == length(constraint_storage)
    # modify b => (I - G) * b + (Gu - g)
    # but Gu = g, so we don't need that here
    # unless we want to modify this to support weakly
    # enforced BCs later
    for i in 1:length(constraint_storage)
        @inbounds b[i] = (1. - constraint_storage[i]) * b[i]
    end
    return nothing
end
  
# COV_EXCL_START
KA.@kernel function _adjust_vector_entries_for_constraints_kernel(b, constraint_storage)
    I = KA.@index(Global)
    # modify b => (I - G) * b + (Gu - g)
    @inbounds b[I] = (1. - constraint_storage[I]) * b[I]
end
# COV_EXCL_STOP
  
function _adjust_vector_entries_for_constraints!(b, constraint_storage, backend::KA.Backend)
    @assert length(b) == length(constraint_storage)
    kernel! = _adjust_vector_entries_for_constraints_kernel(backend)
    kernel!(b, constraint_storage, ndrange = length(b))
    return nothing
end
