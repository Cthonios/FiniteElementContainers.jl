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
# function BCBookKeeping(dof::DofManager, var_name::Symbol, sset_name::Symbol; build_dofs_array=false)
#   # need to extract the var from dof based on teh symbol name
#   var_index = 0
#   dof_index = 0
#   found = false
#   for (vi, var) in enumerate(dof.H1_vars)
#     for name in names(var)
#       dof_index = dof_index + 1
#       if var_name == name
#         var_index = vi
#         found = true
#         break
#       end
#     end
#   end

#   if build_dofs_array
#     @assert found == true "Failed to find variable $var_name"
#   end

#   @assert dof_index <= length(mapreduce(x -> names(x), +, dof.H1_vars)) "Found invalid dof index"

#   # TODO
#   fspace = dof.H1_vars[var_index].fspace
  
#   elems = getproperty(fspace.sideset_elems, sset_name)
#   nodes = getproperty(fspace.sideset_nodes, sset_name)
#   sides = getproperty(fspace.sideset_sides, sset_name)

#   blocks = Vector{Int64}(undef, 0)

#   # gather the blocks that are present in this sideset
#   for (n, val) in enumerate(values(fspace.elem_id_maps))
#     # note these are the local elem id to the block, e.g. starting from 1.
#     indices_in_sset = indexin(val, elems)
#     filter!(x -> x !== nothing, indices_in_sset)
    
#     if length(indices_in_sset) > 0
#       # add stuff to arrays
#       push!(blocks, n)
#     end
#   end

#   # setting up dofs for use in dirichlet bcs
#   if build_dofs_array
#     ND, NN = num_dofs_per_node(dof), num_nodes(dof)
#     n_total_dofs = ND * NN
#     # all_dofs = reshape(1:length(dof), num_dofs_per_node(dof), num_nodes(dof))
#     all_dofs = reshape(1:n_total_dofs, ND, NN)
#     dofs = all_dofs[dof_index, nodes]
#   else
#     dofs = Vector{Int64}(undef, 0)
#   end
  
#   # TODO
#   ND = size(dof.H1_vars[1].fspace.coords, 1)
#   return BCBookKeeping{ND, sset_name, Int64, typeof(blocks)}(blocks, dofs, elems, nodes, sides)
# end

# TODO currently only works for H1 spaces
function BCBookKeeping(dof::DofManager, var_names::Vector{Symbol}, sset_names::Vector{Symbol})
  @assert length(var_names) == length(sset_names)

  elements = Vector{Int}(undef, 0)
  nodes = Vector{Int}(undef, 0)
  sides = Vector{Int}(undef, 0)
  blocks = Vector{Int}(undef, 0)
  dofs = Vector{Int}(undef, 0)

  for (var_name, sset_name) in zip(var_names, sset_names)
    # @info "Reading boundary condition data for variable $var_name on side set $sset_name"
    # see if variable exists
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

    # some error checking
    @assert found == true "Failed to find variable $var_name"
    @assert dof_index <= length(mapreduce(x -> names(x), +, dof.H1_vars)) "Found invalid dof index"
  
    # TODO
    fspace = dof.H1_vars[var_index].fspace

    temp_nodes = getproperty(fspace.sideset_nodes, sset_name)
    append!(elements, getproperty(fspace.sideset_elems, sset_name))
    # append!(nodes, getproperty(fspace.sideset_nodes, sset_name))
    append!(nodes, temp_nodes)
    append!(sides, getproperty(fspace.sideset_sides, sset_name))

    blocks = Vector{Int64}(undef, 0)

    # gather the blocks that are present in this sideset
    # TODO this isn't quite right
    for (n, val) in enumerate(values(fspace.elem_id_maps))
      # note these are the local elem id to the block, e.g. starting from 1.
      indices_in_sset = indexin(val, elements)
      filter!(x -> x !== nothing, indices_in_sset)
      
      if length(indices_in_sset) > 0
        # add stuff to arrays
        push!(blocks, n)
      end
    end

    # setting up dofs for use in dirichlet bcs
    # TODO only works for H1
    ND, NN = num_dofs_per_node(dof), num_nodes(dof)
    n_total_dofs = ND * NN
    # all_dofs = reshape(1:length(dof), num_dofs_per_node(dof), num_nodes(dof))
    all_dofs = reshape(1:n_total_dofs, ND, NN)
    temp_dofs = all_dofs[dof_index, temp_nodes]
    append!(dofs, temp_dofs)
  end

  # dof_perm = _unique_sort_perm(dofs)
  # el_perm = _unique_sort_perm(elements)
  # # TODO fix up blocks first
  # # blocks = blocks[el_perm]
  # dofs = dofs[dof_perm]
  # elements = elements[el_perm]
  # nodes = nodes[dof_perm]
  # sides = sides[el_perm]

  ND = size(dof.H1_vars[1].fspace.coords, 1)

  return BCBookKeeping{ND, :dbcs, Int64, typeof(blocks)}(
    blocks, dofs, elements, nodes, sides
  )
end

function BCBookKeeping(dof::DofManager, var_name::Symbol, sset_name::Symbol)
  return BCBookKeeping(dof, [var_name], [sset_name])
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
# abstract type AbstractBC{
#   S, 
#   # B <: BCBookKeeping, 
#   F <: BCFunction 
# } end
# name(::AbstractBC{S, B, F}) where {S, B, F} = S

abstract type AbstractBC{F} end

function (bc::AbstractBC)(x, t)
  return bc.func(x, t)
end

KA.get_backend(x::AbstractBC) = KA.get_backend(x.vals)

abstract type AbstractBCContainer{
  B <: BCBookKeeping,
  F,
  # I <: AbstractArray{<:Integer, 1},
  V <: AbstractArray{<:Number, 1}
} end

KA.get_backend(x::AbstractBCContainer) = KA.get_backend(x.vals)

# need checks on if field types are compatable
function _update_bc_values!(bc, X, t, ::KA.CPU)
  ND = num_dimensions(bc.bookkeeping)
  for (n, node) in enumerate(bc.bookkeeping.nodes)
    X_temp = @views SVector{ND, eltype(X)}(X[:, node])
    # func_id = bc.func_ids[n]
    # bc.vals[n] = values(bc.funcs)[func_id](X_temp, t)
    bc.vals[n] = bc.func(X_temp, t)
  end
  return nothing
end

KA.@kernel function _update_bc_values_kernel!(bc, X, t)
  I = KA.@index(Global)
  ND = num_dimensions(bc)
  node = bc.bookkeeping.nodes[I]

  # hacky for now, but it works
  # if ND == 1
  #   X_temp = SVector{ND, eltype(X)}(X[1, node])
  # elseif ND == 2
  #   X_temp = SVector{ND, eltype(X)}(X[1, node], X[2, node])
  # elseif ND == 3
  #   X_temp = SVector{ND, eltype(X)}(X[1, node], X[2, node], X[3, node])
  # end
  X_temp = 0.
  # func_id = bc.func_ids[I]
  # # bc.vals[I] = values(bc.funcs)[func_id](X_temp, t)
  # bc.vals[I] = getfield(bc.funcs, func_id)(X_temp, t)
  bc.vals[I] = bc.func(X_temp, t)
end

function _update_bc_values!(bc, X, t, backend::KA.Backend)
  kernel! = _update_bc_values_kernel!(backend)
  kernel!(bc, X, t, ndrange=length(bc.bookkeeping.dofs))
  return nothing
end

# function update_bc_values_old!(bc::AbstractBCContainer, X, t)
#   backend = KA.get_backend(bc)
#   # @assert backend == KA.get_backend(X)
#   _update_bc_values!(bc, X, t, backend)
#   return nothing
# end

function update_bc_values!(bcs, X, t)
  # backend = KA.get_backend(bc)
  for bc in values(bcs)
    backend = KA.get_backend(bc)
    _update_bc_values!(bc, X, t, backend)
  end
  return nothing
end

include("DirichletBCs.jl")
include("NeumannBCs.jl")
