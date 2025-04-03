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
  block_sizes::B
  block_offsets::B
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
function SparsityPattern(dof, type::Type{<:H1Field})

  # get number of dofs for creating cache arrays

  # TODO this line needs to be specialized for aribitrary fields
  # it's hardcoded for H1 firght now.
  ND, NN = num_dofs_per_node(dof), num_nodes(dof)

  n_total_dofs = NN * ND
  vars = _dof_manager_vars(dof, type)
  n_blocks = length(vars[1].fspace.ref_fes)
  # n_blocks = length(dof.H1_vars[1].fspace.ref_fes)

  # first get total number of entries in a stupid manner
  n_entries = 0
  block_sizes = Vector{Int64}(undef, n_blocks)
  block_offsets = Vector{Int64}(undef, n_blocks)

  # for (n, conn) in enumerate(values(dof.H1_vars[1].fspace.elem_conns))
  for (n, conn) in enumerate(values(vars[1].fspace.elem_conns))
    ids = reshape(1:n_total_dofs, ND, NN)
    # conn = reshape(ids[:, conn], ND * size(conn, 1), size(conn, 2))
    # display(block)
    conn = reshape(ids[:, conn], ND * size(conn, 1), size(conn, 2))

    # hacky for now
    # TODO remove this block of code, SVectors are not used
    # for connectivity anymore
    if eltype(conn) <: SVector
      n_entries += length(conn[1])^2 * length(conn)
      block_sizes[n] = length(conn)
      block_offsets[n] = length(conn[1])^2
    else
      n_entries += size(conn, 1)^2 * size(conn, 2)
      block_sizes[n] = size(conn, 2)
      block_offsets[n] = size(conn, 1)^2
    end
  end

  # convert to NamedTuples so it's easy to index
  block_syms = keys(vars[1].fspace.ref_fes)
  block_sizes = NamedTuple{block_syms}(tuple(block_sizes)...)
  block_offsets = NamedTuple{block_syms}(tuple(block_offsets)...)

  # setup pre-allocated arrays based on number of entries found above
  Is = Vector{Int64}(undef, n_entries)
  Js = Vector{Int64}(undef, n_entries)
  unknown_dofs = Vector{Int64}(undef, n_entries)

  # now loop over function spaces and elements
  n = 1
  # for fspace in fspaces
  # for block in valkeys(dof.vars[1].fspace.elem_conns)
  # for conn in values(dof.H1_vars[1].fspace.elem_conns)
  for conn in values(vars[1].fspace.elem_conns)
    ids = reshape(1:n_total_dofs, ND, NN)
    # conn = getproperty(dof.vars[1].fspace.elem_conns, block)
    block_conn = reshape(ids[:, conn], ND * size(conn, 1), size(conn, 2))

    # for e in 1:num_elements(fspace)
    for e in axes(block_conn, 2)
      # conn = dof_connectivity(fspace, e)
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
    block_sizes, block_offsets, 
    # cache arrays
    klasttouch, csrrowptr, csrcolval, csrnzval,
    # additional cache arrays
    csccolptr, cscrowval, cscnzval
  )
end

num_entries(s::SparsityPattern) = length(s.Is)
