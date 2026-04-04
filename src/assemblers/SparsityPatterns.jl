# ndims needs to be 1 (vector) or 2 (matrix)
function _setup_block_sizes(dof::DofManager, ndims::Int)
  ND = size(dof, 1)
  fspace = function_space(dof)
  n_blocks = num_blocks(fspace)
  block_start_indices = Vector{Int64}(undef, n_blocks)

  start_carry = 1
  for b in 1:n_blocks
    NEPE, NE = block_entity_size(fspace, b)
    if ndims == 1
      n_dofs_per_el = ND * NEPE
    elseif ndims == 2
      n_dofs_per_el = (ND * NEPE) * (ND * NEPE)
    end
    block_start_indices[b] = start_carry
    start_carry = start_carry + n_dofs_per_el * NE
  end
  n_entries = start_carry - 1
  return block_start_indices, n_entries
end

# """
# $(TYPEDEF)
# $(TYPEDFIELDS)
# Book-keeping struct for sparse matrices in FEM settings.
# This has all the information to construct a sparse matrix for either
# case where you want to eliminate fixed-dofs or not.
# """
struct SparseMatrixPattern{
  I <: AbstractArray{Int, 1},
  R <: AbstractArray{Float64, 1}
}
  Is::I
  Js::I
  unknown_dofs::I
  block_start_indices::Vector{Int}
  max_entries::Vector{Int}
  # cache arrays
  klasttouch::I
  csrrowptr::I
  csrcolval::I
  csrnzval::R
  # additional cache arrays
  csccolptr::I
  cscrowval::I
  cscnzval::R
  # trying something new
  permutation::I
end

function SparseMatrixPattern(dof::DofManager)

  # get number of dofs for creating cache arrays
  ND, NN = size(dof)
  n_total_dofs = NN * ND

  fspace = function_space(dof)
  n_blocks = num_blocks(fspace)

  block_start_indices, n_entries = _setup_block_sizes(dof, 2)

  # setup pre-allocated arrays based on number of entries found above
  Is = Vector{Int64}(undef, n_entries)
  Js = Vector{Int64}(undef, n_entries)
  unknown_dofs = Vector{Int64}(undef, n_entries)

  # now loop over function spaces and elements
  ids = reshape(1:n_total_dofs, ND, NN)
  n = 1
  for b in 1:n_blocks
    for e in 1:num_elements(fspace, b)
      conn = unsafe_connectivity(fspace, e, b)
      dof_conn = @views reshape(ids[:, conn], ND * num_entities_per_element(fspace, b))
      for i in axes(dof_conn, 1)
        for j in axes(dof_conn, 1)
          Is[n] = dof_conn[i]
          Js[n] = dof_conn[j]
          unknown_dofs[n] = n
          n += 1
        end
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

  # set permutation
  ks = map((i, j) -> (i, j), Is, Js)
  permutation = sortperm(ks)

  pattern = SparseMatrixPattern(
    Is, Js, 
    unknown_dofs, 
    block_start_indices, [n_entries],
    # cache arrays
    klasttouch, csrrowptr, csrcolval, csrnzval,
    # additional cache arrays
    csccolptr, cscrowval, cscnzval,
    permutation
  )

  return pattern
end

function Adapt.adapt_structure(to, asm::SparseMatrixPattern)
  Is = adapt(to, asm.Is)
  Js = adapt(to, asm.Js)
  unknown_dofs = adapt(to, asm.unknown_dofs)
  #
  klasttouch = adapt(to, asm.klasttouch)
  csrrowptr = adapt(to, asm.csrrowptr)
  csrcolval = adapt(to, asm.csrcolval)
  csrnzval = adapt(to, asm.csrnzval)
  #
  csccolptr = adapt(to, asm.csccolptr)
  cscrowval = adapt(to, asm.cscrowval)
  cscnzval = adapt(to, asm.cscnzval)
  perm = adapt(to, asm.permutation)
  return SparseMatrixPattern(
    Is, Js,
    unknown_dofs,
    asm.block_start_indices, asm.max_entries,
    klasttouch, csrrowptr, csrcolval, csrnzval,
    csccolptr, cscrowval, cscnzval, perm
  )
end

function SparseArrays.sparse!(pattern::SparseMatrixPattern, storage)
  return @views SparseArrays.sparse!(
    pattern.Is, pattern.Js, storage[pattern.unknown_dofs],
    length(pattern.klasttouch), length(pattern.klasttouch), +, pattern.klasttouch,
    pattern.csrrowptr, pattern.csrcolval, pattern.csrnzval,
    pattern.csccolptr, pattern.cscrowval, pattern.cscnzval
  )
end

function block_view(storage::AbstractVector, pattern::SparseMatrixPattern, b::Int)
  @assert b > 0 && b <= length(pattern.block_start_indices)
  if b == length(pattern.block_start_indices) || length(pattern.block_start_indices) == 1
    start_index = pattern.block_start_indices[end]
    end_index = length(storage)
  else
    start_index = pattern.block_start_indices[b]
    end_index = pattern.block_start_indices[b + 1]
  end
  indices = start_index:end_index
  return view(storage, indices)
end

num_entries(s::SparseMatrixPattern) = length(s.Is)

# NOTE this methods assumes that dof is up to date
# NOTE this method also only resizes unknown_dofs
# NOTE assumes that dirichlet_dofs comes in sorted and uniqued
# in the pattern object, that means that things
# like Is, Js, etc. need to be viewed into or sliced
function _update_dofs!(pattern::SparseMatrixPattern, dof, dirichlet_dofs)
  n_total_dofs = length(dof) - length(dirichlet_dofs)
  fspace = function_space(dof)

  # create a map for "dof" (e.g. the dof * node for H1) to unknown
  unknown_to_dof = Vector{eltype(dirichlet_dofs)}(undef, n_total_dofs)
  ids = 1:length(dof)
  n = 1
  for dof in ids
    if !insorted(dof, dirichlet_dofs)
      unknown_to_dof[n] = dof
      n += 1
    end
  end
  dof_to_unknown = Dict([(x, y) for (x, y) in zip(unknown_to_dof, 1:length(unknown_to_dof))])
  @assert maximum(values(dof_to_unknown)) == n_total_dofs

  # figure out total number of entries
  ND, NN = size(dof)
  ids = reshape(1:length(dof), ND, NN)
  n_entries = 0
  for b in 1:num_blocks(fspace)
    for e in 1:num_elements(fspace, b)
      conns = unsafe_connectivity(fspace, e, b)
      dof_conns = @views reshape(ids[:, conns], ND * num_entities_per_element(fspace, b))
      for i in axes(dof_conns, 1)
        for j in axes(dof_conns, 1)
          if insorted(dof_conns[i], dirichlet_dofs) || insorted(dof_conns[j], dirichlet_dofs)

          else
            n_entries += 1
          end 
        end
      end
    end
  end

  # # remove me
  # resize!(pattern.Is, 0)
  # resize!(pattern.Js, 0)
  # # end remove me
  # resize!(pattern.unknown_dofs, 0)
  resize!(pattern.Is, n_entries)
  resize!(pattern.Js, n_entries)
  resize!(pattern.unknown_dofs, n_entries)

  # as a safety measure to check later
  fill!(pattern.Is, 0)
  fill!(pattern.Js, 0)
  fill!(pattern.unknown_dofs, 0)

  dof_num = 1
  n = 1
  for b in 1:num_blocks(fspace)
    for e in 1:num_elements(fspace, b)
      conns = unsafe_connectivity(fspace, e, b)
      conn = @views reshape(ids[:, conns], ND * num_entities_per_element(fspace, b))
      for temp in Iterators.product(conn, conn)
        if insorted(temp[1], dirichlet_dofs) || insorted(temp[2], dirichlet_dofs)
          # really do nothing here
        else
          pattern.Is[n] = dof_to_unknown[temp[1]]
          pattern.Js[n] = dof_to_unknown[temp[2]]
          pattern.unknown_dofs[n] = dof_num
          n += 1
        end
        dof_num += 1
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

  # TODO below is half of the runtime in this method
  # it can likely be sped up
  ks = map((i, j) -> (i, j), pattern.Is, pattern.Js)
  permutation = sortperm(ks)
  resize!(pattern.permutation, length(permutation))
  pattern.permutation .= permutation

  return nothing
end

struct SparseVectorPattern{
  I <: AbstractArray{Int, 1}
}
  Is::I
  permutation::I
  unknown_dofs::I
  block_start_indices::Vector{Int}
  max_entries::Vector{Int}
end

function SparseVectorPattern(dof::DofManager)
  ND, NN = size(dof)
  n_total_dofs = NN * ND
  ids = reshape(1:n_total_dofs, ND, NN)

  fspace = function_space(dof)
  block_start_indices, n_entries = _setup_block_sizes(dof, 1)

  # setup pre-allocated arrays based on number of entries
  Is = Vector{Int64}(undef, n_entries)
  unknown_dofs = Vector{Int64}(undef, n_entries)

  n = 1
  for b in 1:num_blocks(fspace)
    conn = connectivity(fspace, b)
    block_conn = reshape(ids[:, conn], ND * size(conn, 1), size(conn, 2))

    for e in axes(block_conn, 2)
      conn = @views block_conn[:, e]
      for temp in conn
        Is[n] = temp
        unknown_dofs[n] = n
        n += 1
      end
    end
  end

  permutation = sortperm(Is)

  return SparseVectorPattern(
    Is, permutation, unknown_dofs,
    block_start_indices, [n_entries]
  )
end

function Adapt.adapt_structure(to, pattern::SparseVectorPattern)
  return SparseVectorPattern(
    adapt(to, pattern.Is),
    adapt(to, pattern.permutation),
    adapt(to, pattern.unknown_dofs),
    pattern.block_start_indices,
    pattern.max_entries
  )
end

function SparseArrays.sparsevec(pattern::SparseVectorPattern, storage)
  return @views sparsevec(pattern.Is, storage[pattern.unknown_dofs])
end

num_entries(pattern::SparseVectorPattern) = length(pattern.Is)

function _update_dofs!(pattern::SparseVectorPattern, dof, dirichlet_dofs)
  n_total_dofs = length(dof) - length(dirichlet_dofs)
  fspace = function_space(dof)

  # create a map for "dof" (e.g. the dof * node for H1) to unknown
  unknown_to_dof = Vector{eltype(dirichlet_dofs)}(undef, n_total_dofs)
  ids = 1:length(dof)
  n = 1
  for dof in ids
    if !insorted(dof, dirichlet_dofs)
      unknown_to_dof[n] = dof
      n += 1
    end
  end
  dof_to_unknown = Dict([(x, y) for (x, y) in zip(unknown_to_dof, 1:length(unknown_to_dof))])
  @assert maximum(values(dof_to_unknown)) == n_total_dofs

  # figure out total number of entries
  ND, NN = size(dof)
  ids = reshape(1:length(dof), ND, NN)
  n_entries = 0
  for b in 1:num_blocks(fspace)
    for e in 1:num_elements(fspace, b)
      conns = unsafe_connectivity(fspace, e, b)
      dof_conns = @views reshape(ids[:, conns], ND * num_entities_per_element(fspace, b))
      for i in axes(dof_conns, 1)
        if insorted(dof_conns[i], dirichlet_dofs)

        else
          n_entries += 1
        end 
      end
    end
  end

  # finish me
  resize!(pattern.Is, n_entries)
  resize!(pattern.unknown_dofs, n_entries)

  fill!(pattern.Is, 0)
  fill!(pattern.unknown_dofs, 0)

  dof_num = 1
  n = 1
  for b in 1:num_blocks(fspace)
    for e in 1:num_elements(fspace, b)
      conns = unsafe_connectivity(fspace, e, b)
      conn = @views reshape(ids[:, conns], ND * num_entities_per_element(fspace, b))
      for temp in conn
        if insorted(temp[1], dirichlet_dofs)
          # really do nothing here
        else
          pattern.Is[n] = dof_to_unknown[temp]
          pattern.unknown_dofs[n] = dof_num
          n += 1
        end
        dof_num += 1
      end
    end
  end

  permutation = sortperm(pattern.Is)
  resize!(pattern.permutation, length(permutation))
  pattern.permutation .= permutation
  return nothing
end
