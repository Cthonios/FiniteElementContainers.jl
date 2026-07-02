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

# below method only makes sense for sparse matrix patterns
function _setup_block_sizes(dof_1::DofManager, dof_2::DofManager)
  ND1, ND2 = size(dof_1, 1), size(dof_2, 1)
  fspace_1, fspace_2 = function_space(dof_1), function_space(dof_2)
  @assert num_blocks(fspace_1) == num_blocks(fspace_2) "Got different number of blocks in the two function spaces."
  n_blocks = num_blocks(fspace_1)
  block_start_indices = Vector{Int64}(undef, n_blocks)
  start_carry = 1
  for b in 1:n_blocks
    NEPE1, NE1 = block_entity_size(fspace_1, b)
    NEPE2, NE2 = block_entity_size(fspace_2, b)
    @assert NE1 == NE2 "Got different numbers of elements in the two function spaces."
    n_dofs_per_el = (ND1 * NEPE1) * (ND2 * NEPE2)
    block_start_indices[b] = start_carry
    start_carry = start_carry + n_dofs_per_el * NE1
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
  cscnzval_mass::R
  cscnzval_stiffness::R
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

  csccolptr          = Vector{Int64}(undef, 0)
  cscrowval          = Vector{Int64}(undef, 0)
  cscnzval_mass      = Vector{Float64}(undef, 0)
  cscnzval_stiffness = Vector{Float64}(undef, 0)

  # set permutation — pack (row, col) into one Int64 key so sortperm takes the
  # integer radix-sort path instead of an O(N log N) tuple comparison sort.
  # DOF indices are << 2^32, so the packed key preserves (i, j) lexicographic
  # order and (sortperm being stable) yields an identical permutation.
  ks = map((i, j) -> (Int64(i) << 32) | Int64(j), Is, Js)
  permutation = sortperm(ks)

  pattern = SparseMatrixPattern(
    Is, Js, 
    unknown_dofs, 
    block_start_indices, [n_entries],
    # cache arrays
    klasttouch, csrrowptr, csrcolval, csrnzval,
    # additional cache arrays
    csccolptr, cscrowval, cscnzval_mass, cscnzval_stiffness,
    permutation
  )

  return pattern
end

function SparseMatrixPattern(dof_1::DofManager, dof_2::DofManager)
  ND1, NN1 = size(dof_1)
  ND2, NN2 = size(dof_2)
  n_total_dofs_1 = NN1 * ND1
  n_total_dofs_2 = NN2 * ND2

  fspace_1, fspace_2 = function_space(dof_1), function_space(dof_2)
  n_blocks = num_blocks(fspace_1)
  block_start_indices, n_entries = _setup_block_sizes(dof_1, dof_2)

  # setup pre-allocated arrays based on number of entries found above
  Is = Vector{Int64}(undef, n_entries)
  Js = Vector{Int64}(undef, n_entries)
  unknown_dofs = Vector{Int64}(undef, n_entries)

  ids_1 = reshape(1:n_total_dofs_1, ND1, NN1)
  ids_2 = reshape(1:n_total_dofs_2, ND2, NN2)
  n = 1
  for b in 1:n_blocks
    for e in 1:num_elements(fspace_1, b)
      conn_1 = unsafe_connectivity(fspace_1, e, b)
      conn_2 = unsafe_connectivity(fspace_2, e, b)
      dof_conn_1 = @views reshape(ids_1[:, conn_1], ND1 * num_entities_per_element(fspace_1, b))
      dof_conn_2 = @views reshape(ids_2[:, conn_2], ND2 * num_entities_per_element(fspace_2, b))
      for i in axes(dof_conn_1, 1)
        for j in axes(dof_conn_2, 1)
          Is[n] = dof_conn_1[i]
          Js[n] = dof_conn_2[j]
          unknown_dofs[n] = n
          n += 1
        end
      end
    end
  end

  # create caches
  # TODO not sure what to do for klasttouch and csrrowptr
  # so making it the maximum size so we don't hit index out of bounds
  # errors
  # Is this right though?
  n_total_dofs = max(n_total_dofs_1, n_total_dofs_2)
  # hack above
  klasttouch = zeros(Int64, n_total_dofs)
  csrrowptr  = zeros(Int64, n_total_dofs + 1)
  csrcolval  = zeros(Int64, length(Is))
  csrnzval   = zeros(Float64, length(Is))

  csccolptr          = Vector{Int64}(undef, 0)
  cscrowval          = Vector{Int64}(undef, 0)
  cscnzval_mass      = Vector{Float64}(undef, 0)
  cscnzval_stiffness = Vector{Float64}(undef, 0)

  # set permutation — pack (row, col) into one Int64 key so sortperm takes the
  # integer radix-sort path instead of an O(N log N) tuple comparison sort.
  # DOF indices are << 2^32, so the packed key preserves (i, j) lexicographic
  # order and (sortperm being stable) yields an identical permutation.
  ks = map((i, j) -> (Int64(i) << 32) | Int64(j), Is, Js)
  permutation = sortperm(ks)

  pattern = SparseMatrixPattern(
    Is, Js, 
    unknown_dofs, 
    block_start_indices, [n_entries],
    # cache arrays
    klasttouch, csrrowptr, csrcolval, csrnzval,
    # additional cache arrays
    csccolptr, cscrowval, cscnzval_mass, cscnzval_stiffness,
    permutation
  )

  return pattern
end

function Adapt.adapt_structure(to, asm::SparseMatrixPattern)
  return SparseMatrixPattern(
    adapt(to, asm.Is),
    adapt(to, asm.Js),
    adapt(to, asm.unknown_dofs),
    asm.block_start_indices, asm.max_entries,
    adapt(to, asm.klasttouch),
    adapt(to, asm.csrrowptr),
    adapt(to, asm.csrcolval),
    adapt(to, asm.csrnzval),
    adapt(to, asm.csccolptr),
    adapt(to, asm.cscrowval),
    adapt(to, asm.cscnzval_mass), adapt(to, asm.cscnzval_stiffness),
    adapt(to, asm.permutation)
  )
end

function KA.get_backend(pattern::SparseMatrixPattern)
  return KA.get_backend(pattern.Is)
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
function _update_dofs!(
  pattern::SparseMatrixPattern,
  dof_1, dirichlet_dofs_1, periodic_side_b_dofs_1,
  dof_2, dirichlet_dofs_2, periodic_side_b_dofs_2
)
  n_total_dofs_1 = length(dof_1) - length(dirichlet_dofs_1) - length(periodic_side_b_dofs_1)
  n_total_dofs_2 = length(dof_2) - length(dirichlet_dofs_2) - length(periodic_side_b_dofs_2)
  fspace_1, fspace_2 = function_space(dof_1), function_space(dof_2)
  @assert num_blocks(fspace_1) == num_blocks(fspace_2)

  # figure out total number of entries
  ND1, NN1 = size(dof_1)
  ND2, NN2 = size(dof_2)

  ids_1 = reshape(1:length(dof_1), ND1, NN1)
  ids_2 = reshape(1:length(dof_2), ND2, NN2)
  
  n_entries = 0
  for b in 1:num_blocks(fspace_1)
    # @assert num_elements(fspace_1) == num_elements(fspace_2)
    @assert block_entity_size(fspace_1, b)[2] == block_entity_size(fspace_2, b)[2]
    for e in 1:num_elements(fspace_1, b)
      conns_1 = unsafe_connectivity(fspace_1, e, b)
      conns_2 = unsafe_connectivity(fspace_2, e, b)
      dof_conns_1 = @views reshape(ids_1[:, conns_1], ND1 * num_entities_per_element(fspace_1, b))
      dof_conns_2 = @views reshape(ids_2[:, conns_2], ND2 * num_entities_per_element(fspace_2, b))
      for i in axes(dof_conns_1, 1)
        ri = dof_to_unknown_index(dof_1, dof_conns_1[i])
        for j in axes(dof_conns_2, 1)
          rj = dof_to_unknown_index(dof_2, dof_conns_2[j])
          if ri > 0 && rj > 0
            n_entries += 1
          end
        end
      end
    end
  end

  resize!(pattern.Is, n_entries)
  resize!(pattern.Js, n_entries)
  resize!(pattern.unknown_dofs, n_entries)

  # as a safety measure to check later
  fill!(pattern.Is, 0)
  fill!(pattern.Js, 0)
  fill!(pattern.unknown_dofs, 0)

  dof_num = 1
  n = 1
  for b in 1:num_blocks(fspace_1)
    for e in 1:num_elements(fspace_1, b)
      conns_1 = unsafe_connectivity(fspace_1, e, b)
      conns_2 = unsafe_connectivity(fspace_2, e, b)
      dof_conns_1 = @views reshape(ids_1[:, conns_1], ND1 * num_entities_per_element(fspace_1, b))
      dof_conns_2 = @views reshape(ids_2[:, conns_2], ND2 * num_entities_per_element(fspace_2, b))
      for i in axes(dof_conns_1, 1)
        ri = dof_to_unknown_index(dof_1, dof_conns_1[i])
        for j in axes(dof_conns_2, 1)
          rj = dof_to_unknown_index(dof_2, dof_conns_2[j])
          if ri > 0 && rj > 0
            pattern.Is[n] = ri
            pattern.Js[n] = rj
            pattern.unknown_dofs[n] = dof_num
            n += 1
          end
          dof_num += 1
        end
      end
    end
  end

  # TODO check this sizing
  n_total_dofs = max(n_total_dofs_1, n_total_dofs_2)
  resize!(pattern.klasttouch, n_total_dofs)
  resize!(pattern.csrrowptr, n_total_dofs + 1)
  # TODO Not sure about below 2 sizes
  resize!(pattern.csrcolval, length(pattern.Is))
  resize!(pattern.csrnzval, length(pattern.Is))

  # Pack (row, col) into one Int64 key so sortperm takes the integer radix-sort
  # path instead of an O(N log N) tuple comparison sort.  DOF indices are
  # << 2^32, so the packed key preserves (i, j) lexicographic order and
  # (sortperm being stable) yields an identical permutation.
  ks = map((i, j) -> (Int64(i) << 32) | Int64(j), pattern.Is, pattern.Js)
  permutation = sortperm(ks)
  resize!(pattern.permutation, length(permutation))
  pattern.permutation .= permutation

  return nothing
end

# some helpers for different sparse types
abstract type AbstractSparseMatrixType end
struct COOMatrix <: AbstractSparseMatrixType
end
struct CSCMatrix <: AbstractSparseMatrixType
end
struct CSRMatrix <: AbstractSparseMatrixType
end

struct UnsupportedSparseMatrixType <: Exception
end

function _sym_to_sparse_matrix_type(sym::Symbol)
  if sym  == :coo
    return COOMatrix()
  elseif sym == :csc
    return CSCMatrix()
  elseif sym == :csr
    return CSRMatrix()
  else
    throw(UnsupportedSparseMatrixType("Unsupported sparse matrix type $sym. Only :coo, :csc, and :csr are supported."))
  end
end

# some low level sparse matrix type helpers
function _coo_matrix(pattern::SparseMatrixPattern, dof, storage)
  backend = KA.get_backend(pattern)
  return _coo_matrix(backend, pattern, dof, storage)
end

function _coo_matrix(backend::KA.Backend, pattern::SparseMatrixPattern, dof, storage)
  constructor = _coo_matrix_constructor(backend)

  if FiniteElementContainers._is_condensed(dof)
    n_dofs = length(dof)
  else
    n_dofs = length(dof.unknown_dofs)
  end

  rows, cols = pattern.Is, pattern.Js
  perm = pattern.permutation
  vals = storage[pattern.unknown_dofs]
  return constructor(
    rows[perm], cols[perm], vals[perm],
    (n_dofs, n_dofs), length(pattern.Is)
  )
end

# TODO could add SparseMatricsCOO.jl as a package to support this
function _coo_matrix(::KA.CPU, ::SparseMatrixPattern, dof, storage)
  @assert false "Currently unsupported"
end

function _coo_matrix_constructor(backend::KA.Backend) 
  @assert false "Need to implement for $backend"
end

function _csc_matrix(pattern::SparseMatrixPattern, dof, coo_storage, csc_storage)
  backend = KA.get_backend(pattern)
  return _csc_matrix(backend, pattern, dof, coo_storage, csc_storage)
end

function _csc_matrix(backend::KA.Backend, pattern::SparseMatrixPattern, dof, coo_storage, csc_storage)
  coo = _coo_matrix(backend, pattern, dof, coo_storage)
  constructor = _csc_matrix_constructor(backend)
  return constructor(coo)
end

function _csc_matrix(::KA.CPU, pattern::SparseMatrixPattern, dof, coo_storage, csc_storage)
  return @views SparseArrays.sparse!(
    pattern.Is, pattern.Js, coo_storage[pattern.unknown_dofs],
    length(pattern.klasttouch), length(pattern.klasttouch), +, pattern.klasttouch,
    pattern.csrrowptr, pattern.csrcolval, pattern.csrnzval,
    pattern.csccolptr, pattern.cscrowval, csc_storage
  )
end

function _csc_matrix_constructor(backend::KA.Backend)
  @assert false "Need to implement for $backend"
end

function _csr_matrix(pattern::SparseMatrixPattern, dof, coo_storage, csc_storage)
  backend = KA.get_backend(pattern)
  return _csr_matrix(backend, pattern, dof, coo_storage, csc_storage)
end

function _csr_matrix(backend::KA.Backend, pattern::SparseMatrixPattern, dof, coo_storage, csc_storage)
  coo = _coo_matrix(backend, pattern, dof, coo_storage)
  constructor = _csr_matrix_constructor(backend)
  return constructor(coo)
end

# TODO eventually write kernels to do this without going to csc first
function _csr_matrix(backend::KA.CPU, pattern::SparseMatrixPattern, dof, coo_storage, csc_storage)
  csc = _csc_matrix(backend, pattern, dof, coo_storage, csc_storage)
  return SparseMatrixCSR(csc)
end

function _csr_matrix_constructor(backend::KA.Backend)
  @assert false "Need to implement for $backend"
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

function _update_dofs!(pattern::SparseVectorPattern, dof, dirichlet_dofs, periodic_side_b_dofs)
  n_total_dofs = length(dof) - length(dirichlet_dofs) - length(periodic_side_b_dofs)
  fspace = function_space(dof)

  # figure out total number of entries
  ND, NN = size(dof)
  ids = reshape(1:length(dof), ND, NN)
  n_entries = 0
  for b in 1:num_blocks(fspace)
    for e in 1:num_elements(fspace, b)
      conns = unsafe_connectivity(fspace, e, b)
      dof_conns = @views reshape(ids[:, conns], ND * num_entities_per_element(fspace, b))
      for i in axes(dof_conns, 1)
        ri = dof_to_unknown_index(dof, dof_conns[i])
        if ri > 0
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
        ri = dof_to_unknown_index(dof, temp)
        if ri > 0
          pattern.Is[n] = ri
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
