abstract type AbstractBCBookKeeping{D, S, T <: Integer, V <: AbstractArray{T, 1}} end

struct BCBookKeeping{D, S, T, V} <: AbstractBCBookKeeping{D, S, T, V}
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
  B <: AbstractBCBookKeeping, 
  F <: BCFunction, 
  V <: AbstractArray{<:Number, 1}
} end
name(::AbstractBC{S, B, F, V}) where {S, B, F, V} = S

# function Base.NamedTuple(bcs::AbstractArray{T, 1}) where T <: AbstractBC
#   syms = map(x -> name(x), bcs)
#   return NamedTuple{syms}(bcs)
# end

function (bc::AbstractBC)(x, t)
  return bc.func(x, t)
end

KA.get_backend(x::AbstractBC) = KA.get_backend(x.vals)

# need checks on if field types are compatable
function _update_bc_values!(bc, X, t, ::KA.CPU)
  ND = num_dimensions(bc.bookkeeping)
  # for (n, dof) in enumerate(bc.bookkeeping.dofs)
  for (n, node) in enumerate(bc.bookkeeping.nodes)
    X_temp = @views SVector{ND, eltype(X)}(X[:, node])
    # bc.vals[n] = bc.func(X_temp, t)
    func_id = bc.func_ids[n]
    bc.vals[n] = values(bc.funcs)[func_id](X_temp, t)
  end
  return nothing
end

KA.@kernel function _update_bc_values_kernel!(bc, X, t)
  I = KA.@index(Global)
  # ND = num_dimensions(bc)
  # dof = bc.bookkeeping.dofs[I]
  # X_temp = SVector{ND, eltype(X)}(X[:, dof])
  # X_temp = X[:, bc.bookkeeping.dofs[I]]
  # @inbounds bc.vals[I] = bc.func(X[:, bc.bookkeeping.dofs[I]], t)
  # TODO can't get the coordinates to propagate through
  # bc.vals[I] = bc.func(0., t)
  func_id = bc.func_ids[I]
  bc.vals[I] = values(bc.funcs)[func_id](0., t)
end

function _update_bc_values!(bc, X, t, backend::KA.Backend)
  kernel! = _update_bc_values_kernel!(backend)
  kernel!(bc, X, t, ndrange=length(bc.bookkeeping.dofs))
  return nothing
end

function update_bc_values!(bc, X, t)
  backend = KA.get_backend(bc)
  @assert backend == KA.get_backend(X)
  _update_bc_values!(bc, X, t, backend)
  return nothing
end

# function nonunique(v)
#   sv = sort(v)
#   return unique(@view sv[[diff(sv).==0; false]])
# end

# struct DirichletBCCollection{Fs, IDs}
#   funcs::Fs
#   func_ids::IDs
# end

# # Better to iterate over functions and feed each func into a kernel
# # with the associated local to global Ubc IDs

# function DirichletBCCollection(dbcs)
#   # setup func nt
#   funcs = tuple(map(x -> x.func, dbcs)...)
#   func_names = tuple(map(x -> Symbol("func_$x"), 1:length(funcs))...)
#   funcs = NamedTuple{func_names}(funcs)

#   dofs = tuple(map(x -> x.bookkeeping.dofs, dbcs)...)
#   n_dofs = length(unique(vcat(dofs...)))

#   n_per_func = map(x -> length(x.bookkeeping.dofs))

#   # grab all dofs
#   # dofs = mapreduce(x -> x.bookkeeping.dofs, vcat, dbcs)
#   # dofs_perm = sortperm(dofs)
#   # dofs_unique = unique(i -> dofs[i], eachindex(dofs))

#   # # dofs = nonunique(mapreduce(x -> x.bookkeeping.dofs, vcat, dbcs))
#   # for dof in nonunique(dofs)
#   #   @warn "Repeated dof $dof encounted in DirichletBCCollection"
#   # end

#   # func_ids = zeros(Int, 0)
#   # for (n, bc) in enumerate(dbcs)
#   #   # append!(func_ids, )
#   #   append!(func_ids, fill(n, length(bc.bookkeeping.dofs)))
#   # end

  

#   # func_ids = func_ids[dofs_perm][dofs_unique]
#   # return DirichletBCCollection(funcs, func_ids)
# end

# function create_bcs(dbcs::DirichletBCCollection, time)
#   backend = KA.get_backend(dbcs.func_ids)
#   vals = KA.zeros(backend, Float64, length(dbcs.func_ids))

#   AK.foreachindex(dbcs.func_ids) do n
#     # TODO change to coords
#     func_id = dbcs.func_ids[n]
#     func = values(dbcs.funcs)[func_id]  
#     # vals[n] = values(dbcs.funcs)[dbcs.func_ids[n]](time, time)
#   end
#   return vals
# end

include("DirichletBCs.jl")
include("NeumannBCs.jl")

# CPU only for now
# TODO remove this one eventually
# currenlty used in a mechanics test
function update_field_bcs!(U::H1Field, dof::DofManager, dbc::DirichletBC, t)
  X_global = dof.H1_vars[1].fspace.coords
  for (n, node) in enumerate(dbc.bookkeeping.nodes)
    X = @views X_global[:, node]
    dbc.vals[n] = dbc.func(X, t)
  end
  for (n, dof) in enumerate(dbc.bookkeeping.dofs)
    U[dof] = dbc.vals[n]
  end
  return nothing
end

