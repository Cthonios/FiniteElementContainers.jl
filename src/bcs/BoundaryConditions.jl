function _unique_sort_perm(array::AbstractArray{T, 1}) where T <: Number
  # chatgpt BS below. Make this not use a dict
  sorted_unique = sort(unique(array))
  id_map = Dict(x => findfirst(==(x), array) for x in sorted_unique)
  return [id_map[x] for x in sorted_unique]
end

struct BCBookKeeping{D, S, T, V}
  blocks::V
  dofs::V
  elements::V
  nodes::V
  sides::V
end

# TODO hardcoded for H1 spaces right now.

# TODO also need to adapt this to differ on what var_name we look for based on build_dofs_array
# e.g. if it's neumann and a vector look for :u but if it's dirichlet and a vector look for :u_x
# this only works for H1 spaces currently
function BCBookKeeping(dof::DofManager, var_name::Symbol, sset_name::Symbol; build_dofs_array=false)
  # need to extract the var from dof based on teh symbol name
  var_index = 0
  dof_index = 0
  found = false
  for (vi, var) in enumerate(dof.H1_vars)
    for name in names(var)
      dof_index = dof_index + 1
      if var_name == name
        var_index = vi
        found = true
        break
      end
    end
  end

  if build_dofs_array
    @assert found == true "Failed to find variable $var_name"
  end

  @assert dof_index <= length(mapreduce(x -> names(x), +, dof.H1_vars)) "Found invalid dof index"

  # TODO
  fspace = dof.H1_vars[var_index].fspace
  
  elems = getproperty(fspace.sideset_elems, sset_name)
  nodes = getproperty(fspace.sideset_nodes, sset_name)
  sides = getproperty(fspace.sideset_sides, sset_name)

  blocks = Vector{Int64}(undef, 0)

  # gather the blocks that are present in this sideset
  for (n, val) in enumerate(values(fspace.elem_id_maps))
    # note these are the local elem id to the block, e.g. starting from 1.
    indices_in_sset = indexin(val, elems)
    filter!(x -> x !== nothing, indices_in_sset)
    
    if length(indices_in_sset) > 0
      # add stuff to arrays
      push!(blocks, n)
    end
  end

  # setting up dofs for use in dirichlet bcs
  if build_dofs_array
    ND, NN = num_dofs_per_node(dof), num_nodes(dof)
    n_total_dofs = ND * NN
    # all_dofs = reshape(1:length(dof), num_dofs_per_node(dof), num_nodes(dof))
    all_dofs = reshape(1:n_total_dofs, ND, NN)
    dofs = all_dofs[dof_index, nodes]
  else
    dofs = Vector{Int64}(undef, 0)
  end
  
  # TODO
  ND = size(dof.H1_vars[1].fspace.coords, 1)
  return BCBookKeeping{ND, sset_name, Int64, typeof(blocks)}(blocks, dofs, elems, nodes, sides)
end

KA.get_backend(bk::BCBookKeeping) = KA.get_backend(bk.blocks)
num_dimensions(::BCBookKeeping{D, S, T, V}) where {D, S, T, V} = D

struct BCFunction{T}
  func::T
end

function (bc::BCFunction{T})(x, t) where T
  return bc.func(x, t)
end

# actual bcs
abstract type AbstractBC{
  S, 
  B <: BCBookKeeping, 
  F <: BCFunction 
} end
name(::AbstractBC{S, B, F}) where {S, B, F} = S

function (bc::AbstractBC)(x, t)
  return bc.func(x, t)
end

KA.get_backend(x::AbstractBC) = KA.get_backend(x.vals)

abstract type AbstractBCContainer{
  B <: BCBookKeeping,
  F,
  I <: AbstractArray{<:Integer, 1},
  V <: AbstractArray{<:Number, 1}
} end

KA.get_backend(x::AbstractBCContainer) = KA.get_backend(x.vals)

# need checks on if field types are compatable
function _update_bc_values!(bc, X, t, ::KA.CPU)
  ND = num_dimensions(bc.bookkeeping)
  for (n, node) in enumerate(bc.bookkeeping.nodes)
    X_temp = @views SVector{ND, eltype(X)}(X[:, node])
    func_id = bc.func_ids[n]
    bc.vals[n] = values(bc.funcs)[func_id](X_temp, t)
  end
  return nothing
end

KA.@kernel function _update_bc_values_kernel!(bc, X, t)
  I = KA.@index(Global)
  ND = num_dimensions(bc)
  node = bc.bookkeeping.nodes[I]

  # hacky for now, but it works
  if ND == 1
    X_temp = SVector{ND, eltype(X)}(X[1, node])
  elseif ND == 2
    X_temp = SVector{ND, eltype(X)}(X[1, node], X[2, node])
  elseif ND == 3
    X_temp = SVector{ND, eltype(X)}(X[1, node], X[2, node], X[3, node])
  end

  func_id = bc.func_ids[I]
  bc.vals[I] = values(bc.funcs)[func_id](X_temp, t)
end

function _update_bc_values!(bc, X, t, backend::KA.Backend)
  kernel! = _update_bc_values_kernel!(backend)
  kernel!(bc, X, t, ndrange=length(bc.bookkeeping.dofs))
  return nothing
end

function update_bc_values!(bc::AbstractBCContainer, X, t)
  backend = KA.get_backend(bc)
  @assert backend == KA.get_backend(X)
  _update_bc_values!(bc, X, t, backend)
  return nothing
end

include("DirichletBCs.jl")
include("NeumannBCs.jl")
