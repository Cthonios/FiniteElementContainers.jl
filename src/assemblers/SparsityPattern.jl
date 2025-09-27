"""
$(TYPEDEF)
$(TYPEDFIELDS)
Book-keeping struct for sparse matrices in FEM settings.
This has all the information to construct a sparse matrix for either
case where you want to eliminate fixed-dofs or not.
"""
struct SparsityPattern{
  I <: AbstractArray{Int, 1},
  B,
  R <: AbstractArray{Float64, 1}
}
  Is::I
  Js::I
  unknown_dofs::I
  block_start_indices::B
  block_el_level_sizes::B
  # cache arrays
  klasttouch::I
  csrrowptr::I
  csrcolval::I
  csrnzval::R
  # additional cache arrays
  csccolptr::I
  cscrowval::I
  cscnzval::R
end

# TODO won't work for H(div) or H(curl) yet
function SparsityPattern(dof::DofManager)

  # get number of dofs for creating cache arrays

  # TODO this line needs to be specialized for aribitrary fields
  # it's hardcoded for H1 firght now.
  # ND, NN = num_dofs_per_node(dof), num_nodes(dof)
  ND, NN = size(dof)
  n_total_dofs = NN * ND

  # vars = _dof_manager_vars(dof, type)
  # n_blocks = length(vars[1].fspace.ref_fes)
  # n_blocks = length(dof.H1_vars[1].fspace.ref_fes)

  fspace = function_space(dof)
  n_blocks = length(fspace.ref_fes)

  # first get total number of entries in a stupid manner
  n_entries = 0
  block_start_indices = Vector{Int64}(undef, n_blocks)
  block_el_level_sizes = Vector{Int64}(undef, n_blocks)

  # for (n, conn) in enumerate(values(dof.H1_vars[1].fspace.elem_conns))
  start_carry = 1
  for (n, conn) in enumerate(values(fspace.elem_conns))
    ids = reshape(1:n_total_dofs, ND, NN)
    # TODO do we need this operation?
    conn = reshape(ids[:, conn], ND * size(conn, 1), size(conn, 2))
    n_entries += size(conn, 1)^2 * size(conn, 2)

    # block_start_indices[n] = size(conn, 2)
    # if n == 1
    #   block_start_indices[n] = 1
    # else
    #   block_start_indices[n] = carry
    # end
    block_start_indices[n] = start_carry
  
    start_carry = start_carry + size(conn, 1)^2 * size(conn, 2)
    block_el_level_sizes[n] = size(conn, 1)^2
  end

  # convert to NamedTuples so it's easy to index
  block_syms = keys(fspace.ref_fes)
  block_start_indices = NamedTuple{block_syms}(tuple(block_start_indices)...)
  block_el_level_sizes = NamedTuple{block_syms}(tuple(block_el_level_sizes)...)

  # setup pre-allocated arrays based on number of entries found above
  Is = Vector{Int64}(undef, n_entries)
  Js = Vector{Int64}(undef, n_entries)
  unknown_dofs = Vector{Int64}(undef, n_entries)

  # now loop over function spaces and elements
  n = 1
  for conn in values(fspace.elem_conns)
    ids = reshape(1:n_total_dofs, ND, NN)
    # TODO do we need this?
    block_conn = reshape(ids[:, conn], ND * size(conn, 1), size(conn, 2))

    for e in axes(block_conn, 2)
      conn = @views block_conn[:, e]
      for temp in Iterators.product(conn, conn)
        Is[n] = temp[1]
        Js[n] = temp[2]
        unknown_dofs[n] = n
        n += 1
      end
    end
  end

  # create caches
  klasttouch = zeros(Int64, n_total_dofs)
  csrrowptr  = zeros(Int64, n_total_dofs + 1)
  csrcolval  = zeros(Int64, length(Is))
  csrnzval   = zeros(Float64, length(Is))

  csccolptr = Vector{Int64}(undef, 0)
  cscrowval = Vector{Int64}(undef, 0)
  cscnzval  = Vector{Float64}(undef, 0)

  return SparsityPattern(
    Is, Js, 
    unknown_dofs, 
    block_start_indices, block_el_level_sizes, 
    # cache arrays
    klasttouch, csrrowptr, csrcolval, csrnzval,
    # additional cache arrays
    csccolptr, cscrowval, cscnzval
  )
end

function SparseArrays.sparse!(pattern::SparsityPattern, storage)
  return @views SparseArrays.sparse!(
    pattern.Is, pattern.Js, storage[pattern.unknown_dofs],
    length(pattern.klasttouch), length(pattern.klasttouch), +, pattern.klasttouch,
    pattern.csrrowptr, pattern.csrcolval, pattern.csrnzval,
    pattern.csccolptr, pattern.cscrowval, pattern.cscnzval
  )
end

num_entries(s::SparsityPattern) = length(s.Is)

# NOTE this methods assumes that dof is up to date
# NOTE this method also only resizes unknown_dofs
# in the pattern object, that means that things
# like Is, Js, etc. need to be viewed into or sliced
function _update_dofs!(pattern::SparsityPattern, dof, dirichlet_dofs)
  n_total_dofs = length(dof) - length(dirichlet_dofs)

  # remove me
  resize!(pattern.Is, 0)
  resize!(pattern.Js, 0)
  # end remove me
  resize!(pattern.unknown_dofs, 0)
  ND, NN = size(dof)
  ids = reshape(1:length(dof), ND, NN)
  fspace = function_space(dof)

  n = 1
  for conns in values(fspace.elem_conns)
    dof_conns = @views reshape(ids[:, conns], ND * size(conns, 1), size(conns, 2))

    for e in 1:size(conns, 2)
      conn = @views dof_conns[:, e]
      for temp in Iterators.product(conn, conn)
        if insorted(temp[1], dirichlet_dofs) || insorted(temp[2], dirichlet_dofs)
          # really do nothing here
        else
          # remove me
          push!(pattern.Is, temp[1] - count(x -> x < temp[1], dirichlet_dofs))
          push!(pattern.Js, temp[2] - count(x -> x < temp[2], dirichlet_dofs))
          # end remove me
          push!(pattern.unknown_dofs, n)
        end
        n += 1
      end
    end
  end

  resize!(pattern.klasttouch, n_total_dofs)
  resize!(pattern.csrrowptr, n_total_dofs + 1)
  # TODO Not sure about below 2 sizes
  # resize!(assembler.pattern.csrcolval, length(assembler.pattern.Is))
  # resize!(assembler.pattern.csrnzval, length(assembler.pattern.Is))
  resize!(pattern.csrcolval, length(pattern.Is))
  resize!(pattern.csrnzval, length(pattern.Is))

  return nothing
end
