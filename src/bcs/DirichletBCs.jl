abstract type AbstractDirichletBC{F} <: AbstractBC{F} end

const EntityName = Union{Nothing, String}

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
User facing API to define a ```DirichletBC````.
"""
struct DirichletBC{F} <: AbstractDirichletBC{F}
  func::F
  block_name::EntityName
  nset_name::EntityName
  sset_name::EntityName
  var_name::String

  """
  $(TYPEDEF)
  $(TYPEDSIGNATURES)
  $(TYPEDFIELDS)
  """
  function DirichletBC(
    var_name::String, func::Function;
    block_name::Union{Nothing, String} = nothing,
    nodeset_name::Union{Nothing, String} = nothing,
    sideset_name::Union{Nothing, String} = nothing
  )
    if block_name === nothing && nodeset_name === nothing && sideset_name === nothing
      _entity_not_provided_error(
        "block_name, nodeset_name, or sideset_name required" *
        " as input arguments in DirichletBC"
      )
    end
    count = (block_name !== nothing) +
            (nodeset_name !== nothing) +
            (sideset_name !== nothing)
    if count != 1
      _unsure_entity_type_error("More than one entity type specificed in DirichletBC")
    end
    new{typeof(func)}(func, block_name, nodeset_name, sideset_name, var_name)
  end
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
Internal implementation of dirichlet BCs
"""
struct DirichletBCContainer{
  IV <: AbstractArray{<:Integer, 1},
  RV <: AbstractArray{<:Number, 1}
} <: AbstractBCContainer{IV, RV}
  dofs::IV
  nodes::IV
  vals::RV
  vals_dot::RV
  vals_dot_dot::RV

  """
  $(TYPEDEF)
  $(TYPEDSIGNATURES)
  $(TYPEDFIELDS)
  """
  function DirichletBCContainer(mesh, dof::DofManager, dbc::DirichletBC)
    dof_index = _dof_index_from_var_name(dof, dbc.var_name)
    all_dofs = reshape(1:length(dof), size(dof))

    if dbc.block_name !== nothing
      conns = mesh.element_conns[dbc.block_name]
      nodes = sort(unique(conns))
    elseif dbc.nset_name !== nothing
      nodes = mesh.nodeset_nodes[dbc.nset_name]
    elseif dbc.sset_name !== nothing
      nodes = mesh.sideset_nodes[dbc.sset_name]
    end

    # gather dofs associated with nodes
    dofs = all_dofs[dof_index, nodes]

    # sort nodes and dofs for dirichlet bc
    dof_perm = _unique_sort_perm(dofs)
    dofs = dofs[dof_perm]
    nodes = nodes[dof_perm]

    vals = zeros(length(nodes))
    vals_dot = zeros(length(nodes))
    vals_dot_dot = zeros(length(nodes))
    return DirichletBCContainer(dofs, nodes, vals, vals_dot, vals_dot_dot)
  end

  function DirichletBCContainer(dofs, nodes, vals, vals_dot, vals_dot_dot)
    new{typeof(dofs), typeof(vals)}(dofs, nodes, vals, vals_dot, vals_dot_dot)
  end
end

function Adapt.adapt_structure(to, bc::DirichletBCContainer)
  dofs = adapt(to, bc.dofs)
  nodes = adapt(to, bc.nodes)
  vals = adapt(to, bc.vals)
  vals_dot = adapt(to, bc.vals_dot)
  vals_dot_dot = adapt(to, bc.vals_dot_dot)
  return DirichletBCContainer(dofs, nodes, vals, vals_dot, vals_dot_dot)
end

function Base.show(io::IO, bc::DirichletBCContainer)
  println(io, "$(typeof(bc).name.name):")
  println(io, "  Number of active dofs     = $(length(bc.dofs))")
  println(io, "  Number of active nodes    = $(length(bc.nodes))")
end

# this will break whatever is in vals and zero things out
function fuse_bcs(bc1::DirichletBCContainer, args...; sort_and_unique::Bool = true)
  dofs = bc1.dofs
  nodes = bc1.nodes
  for bc2 in args
    append!(dofs, bc2.dofs)
    append!(nodes, bc2.nodes)
  end

  if sort_and_unique
    perm = _unique_sort_perm(dofs)
    dofs = dofs[perm]
    nodes = nodes[perm]
  end

  vals = zeros(eltype(bc1.vals), length(dofs))
  vals_dot = zeros(eltype(bc1.vals_dot), length(dofs))
  vals_dot_dot = zeros(eltype(bc1.vals_dot_dot), length(dofs))
  return DirichletBCContainer(dofs, nodes, vals, vals_dot, vals_dot_dot)
end

struct DirichletBCFunction{F1, F2, F3} <: AbstractBCFunction{F1}
  func::F1
  func_dot::F2
  func_dot_dot::F3
end

function DirichletBCFunction(func)
  func_dot = (x, t) -> ForwardDiff.derivative(z -> func(x, z), t)
  func_dot_dot = (x, t) -> ForwardDiff.derivative(z -> func_dot(x, z), t)
  return DirichletBCFunction(func, func_dot, func_dot_dot)
end

"""
$(TYPEDSIGNATURES)
"""
function _update_bc_values!(bc::DirichletBCContainer, func::DirichletBCFunction, X, t, bc_length::Int, offset::Int)
  backend = KA.get_backend(bc)
  fec_foreach(1:bc_length, backend) do n
    ND = num_fields(X)
    index = n + offset
    node = bc.nodes[index]
  
    # hacky for now, but it works
    # can't do X[:, node] on the GPU, this results in a dynamic
    # function call
    if ND == 1
      X_temp = SVector{ND, eltype(X)}(X[1, node])
    elseif ND == 2
      X_temp = SVector{ND, eltype(X)}(X[1, node], X[2, node])
    elseif ND == 3
      X_temp = SVector{ND, eltype(X)}(X[1, node], X[2, node], X[3, node])
    end

    bc.vals[index] = func.func(X_temp, t)
    bc.vals_dot[index] = func.func_dot(X_temp, t)
    bc.vals_dot_dot[index] = func.func_dot_dot(X_temp, t)
  end
end

struct DirichletBCs{
  BCFuncs,
  IV      <: AbstractArray{<:Integer, 1},
  RV      <: AbstractArray{<:Number, 1},
} <: AbstractBCs{BCFuncs}
  bc_cache::DirichletBCContainer{IV, RV}
  bc_funcs::BCFuncs
  bc_lengths::Vector{Int}

  function DirichletBCs(mesh, dof, bcs_input)
    # base case return empty stuff
    if length(bcs_input) == 0
      bc_cache = DirichletBCContainer(
        zeros(Int, 0), zeros(Int, 0), zeros(Float64, 0), zeros(Float64, 0), zeros(Float64, 0)
      )
      bc_funcs = DirichletBCFunction[]
      return new{Vector{DirichletBCFunction}, Vector{Int}, Vector{Float64}}(
        bc_cache, bc_funcs, Int[]
      )
    end

    # wrap funcs in our DirichletBCFunction
    funcs = map(x -> DirichletBCFunction(x.func), bcs_input)

    # TODO can we make it cleaner so we don't need to fuse
    # and just query mesh for length info?
    caches = DirichletBCContainer.((mesh,), (dof,), bcs_input)

    # fusing bcs with a common function
    func_to_ids = Dict{DirichletBCFunction, Vector{Int}}()
    for (n, func) in enumerate(funcs)
      if haskey(func_to_ids, func)
        push!(func_to_ids[func], n)
      else
        func_to_ids[func] = [n]
      end
    end

    bc_cache = typeof(caches[1])[]
    bc_funcs = DirichletBCFunction[]
    for (func, ids) in pairs(func_to_ids)
      push!(bc_funcs, func)
      push!(bc_cache, fuse_bcs(caches[ids]...))
    end

    # get all lengths for indexing purposes later
    bc_lengths = map(x -> length(x.dofs), bc_cache)

    # now we'll fuse bcs one more time so we only have a single
    # container for all dirichlet bcs. We just use the indexes
    # for dispatching to individual kernels for the different functions
    # TODO the setup could be made more efficient
    # make sure not to sort and unique
    bc_cache = fuse_bcs(bc_cache...; sort_and_unique = false)

    # finally update dofs in dof based on bcs
    temp_dofs = unique(sort(bc_cache.dofs))
    update_dofs!(dof, temp_dofs)

    # return DirichletBCs(bc_cache, bc_funcs, bc_lengths)
    new{typeof(bc_funcs), typeof(bc_lengths), typeof(bc_cache.vals)}(
      bc_cache, bc_funcs, bc_lengths
    )
  end

  function DirichletBCs{BCF, IV, RV}(bc_cache, bc_funcs, bc_lengths) where {BCF, IV, RV}
    new{BCF, IV, RV}(bc_cache, bc_funcs, bc_lengths)
  end
end

function Adapt.adapt_structure(to, bcs::DirichletBCs{BCF, IV, RV}) where {BCF, IV, RV}
  bc_cache = adapt(to, bcs.bc_cache)
  return DirichletBCs{BCF, typeof(bc_cache.dofs), typeof(bc_cache.vals)}(
    bc_cache,
    bcs.bc_funcs,
    bcs.bc_lengths
  )
end

Base.length(bcs::DirichletBCs) = length(bcs.bc_funcs)

# TODO improve this guy
function Base.show(io::IO, bcs::DirichletBCs)
  # for (n, (cache, func)) in enumerate(zip(bcs.bc_cache, bcs.bc_funcs))
  show(io, "DirichletBC:")
  show(io, bcs.bc_cache)
  show(io, bcs.bc_lengths)
  # show(io, func)
  show(io, "\n")
  # end
end

function dirichlet_dofs(bcs::DirichletBCs)
  return unique(sort(bcs.bc_cache.dofs))
end

function update_bc_values!(bcs::DirichletBCs, X, t)
  offset = 0
  for n in axes(bcs.bc_funcs, 1)
    func = bcs.bc_funcs[n]
    bc_length = bcs.bc_lengths[n]
    _update_bc_values!(bcs.bc_cache, func, X, t, bc_length, offset)
    offset += bc_length
  end
  return nothing
end

function update_field_dirichlet_bcs!(U, bcs::DirichletBCs)
  cache = bcs.bc_cache
  fec_foreach(cache.dofs) do I
    dof = cache.dofs[I]
    U[dof] = cache.vals[I]
  end
  return nothing
end

function update_field_dirichlet_bcs!(U, V, bcs::DirichletBCs)
  cache = bcs.bc_cache
  fec_foreach(cache.dofs) do I
    dof = cache.dofs[I]
    U[dof] = cache.vals[I]
    V[dof] = cache.vals_dot[I]
  end
  return nothing
end

function update_field_dirichlet_bcs!(U, V, A, bcs::DirichletBCs)
  cache = bcs.bc_cache
  fec_foreach(cache.dofs) do I
    dof = cache.dofs[I]
    U[dof] = cache.vals[I]
    V[dof] = cache.vals_dot[I]
    A[dof] = cache.vals_dot_dot[I]
  end
  return nothing
end
