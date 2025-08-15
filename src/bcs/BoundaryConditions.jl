function _unique_sort_perm(array::AbstractArray{T, 1}) where T <: Number
  # chatgpt BS below. Make this not use a dict
  sorted_unique = sort(unique(array))
  id_map = Dict(x => findfirst(==(x), array) for x in sorted_unique)
  return [id_map[x] for x in sorted_unique]
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
This struct is used to help with book keeping nodes, sides, etc.
for all types of boundary conditions.

TODO need to add a domain ID for extending to Schwarz
"""
struct BCBookKeeping{
  M <: AbstractArray{<:Integer, 2},
  V <: AbstractArray{<:Integer, 1}
}
  blocks::V
  dofs::V
  elements::V
  nodes::V
  sides::V
  side_nodes::M
end

# TODO currently only works for H1 spaces
# """
# $(TYPEDSIGNATURES)
# """
# function BCBookKeeping_old(
#   dof::DofManager, var_names::Vector{Symbol}, sset_names::Vector{Symbol}
# )
#   @assert length(var_names) == length(sset_names)

#   elements = Vector{Int}(undef, 0)
#   nodes = Vector{Int}(undef, 0)
#   sides = Vector{Int}(undef, 0)
#   side_nodes = Vector{Int}(undef, 0)
#   blocks = Vector{Int}(undef, 0)
#   dofs = Vector{Int}(undef, 0)

#   for (var_name, sset_name) in zip(var_names, sset_names)
#     # @info "Reading boundary condition data for variable $var_name on side set $sset_name"
#     # see if variable exists
#     var_index = 0
#     dof_index = 0
#     found = false
#     for (vi, var) in enumerate(dof.H1_vars)
#       for name in names(var)
#         dof_index = dof_index + 1
#         if var_name == name
#           var_index = vi
#           found = true
#           break
#         end
#       end
#     end

#     # some error checking
#     # @assert found == true "Failed to find variable $var_name"
#     @assert dof_index <= length(mapreduce(x -> names(x), +, dof.H1_vars)) "Found invalid dof index"
  
#     # TODO
#     # fspace = dof.H1_vars[var_index].fspace
#     fspace = function_space(dof, H1Field)

#     temp_nodes = getproperty(fspace.sideset_nodes, sset_name)
#     append!(elements, getproperty(fspace.sideset_elems, sset_name))
#     # append!(nodes, getproperty(fspace.sideset_nodes, sset_name))
#     append!(nodes, temp_nodes)
#     append!(sides, getproperty(fspace.sideset_sides, sset_name))
#     append!(side_nodes, getproperty(fspace.sideset_side_nodes, sset_name))

#     blocks = Vector{Int64}(undef, 0)

#     # gather the blocks that are present in this sideset
#     # TODO this isn't quite right
#     for (n, val) in enumerate(values(fspace.elem_id_maps))
#       # note these are the local elem id to the block, e.g. starting from 1.
#       indices_in_sset = indexin(val, elements)
#       filter!(x -> x !== nothing, indices_in_sset)
      
#       if length(indices_in_sset) > 0
#         append!(blocks, repeat([n], length(indices_in_sset)))
#       end
#     end

#     # setting up dofs for use in dirichlet bcs
#     # TODO only works for H1
#     ND, NN = num_dofs_per_node(dof), num_nodes(dof)
#     n_total_dofs = ND * NN
#     all_dofs = reshape(1:n_total_dofs, ND, NN)
#     temp_dofs = all_dofs[dof_index, temp_nodes]
#     append!(dofs, temp_dofs)
#   end

#   # now sort and unique this stuff
#   dof_perm = _unique_sort_perm(dofs)
#   el_perm = _unique_sort_perm(elements)

#   # do permutations
#   # TODO need to do this in different BC types
#   # since different BC types might want things
#   # organized differently
#   blocks_new = blocks[el_perm]
#   dofs_new = dofs[dof_perm]
#   elements_new = elements[el_perm]
#   nodes_new = nodes[dof_perm]
#   sides_new = sides[el_perm]
#   side_nodes_new = side_nodes[:, el_perm]
#   resize!(blocks, length(blocks_new))
#   resize!(dofs, length(dofs_new))
#   resize!(elements, length(elements_new))
#   resize!(nodes, length(nodes_new))
#   resize!(sides, length(sides_new))
#   resize!(side_nodes, size(side_nodes_new))
#   copyto!(blocks, blocks_new)
#   copyto!(dofs, dofs_new)
#   copyto!(elements, elements_new)
#   copyto!(nodes, nodes_new)
#   copyto!(sides, sides_new)
#   copyto!(side_nodes, side_nodes_new)

#   return BCBookKeeping{typeof(blocks), typeof(side_nodes)}(
#     blocks, dofs, elements, nodes, sides, side_nodes
#   )
# end

# TODO currently only works for H1 spaces
"""
$(TYPEDSIGNATURES)
"""
function BCBookKeeping(
  dof::DofManager, var_name::Symbol, sset_name::Symbol
)
  # check if var exists
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
  # @assert found == true "Failed to find variable $var_name"
  @assert dof_index <= length(mapreduce(x -> names(x), +, dof.H1_vars)) "Found invalid dof index"

  # TODO
  # fspace = dof.H1_vars[var_index].fspace
  fspace = function_space(dof, H1Field)

  # get sset specific fields
  elements = getproperty(fspace.sideset_elems, sset_name)
  nodes = getproperty(fspace.sideset_nodes, sset_name)
  sides = getproperty(fspace.sideset_sides, sset_name)
  side_nodes = getproperty(fspace.sideset_side_nodes, sset_name)

  dofs = Vector{Int64}(undef, 0)
  blocks = Vector{Int64}(undef, 0)

  # gather the blocks that are present in this sideset
  # TODO this isn't quite right
  for (n, val) in enumerate(values(fspace.elem_id_maps))
    # note these are the local elem id to the block, e.g. starting from 1.
    indices_in_sset = indexin(val, elements)
    filter!(x -> x !== nothing, indices_in_sset)
    
    if length(indices_in_sset) > 0
      append!(blocks, repeat([n], length(indices_in_sset)))
    end
  end

  # setting up dofs for use in dirichlet bcs
  # TODO only works for H1
  ND, NN = num_dofs_per_node(dof), num_nodes(dof)
  n_total_dofs = ND * NN
  all_dofs = reshape(1:n_total_dofs, ND, NN)
  # temp_dofs = all_dofs[dof_index, temp_nodes]
  temp_dofs = all_dofs[dof_index, nodes]
  append!(dofs, temp_dofs)
  # end

  # now sort and unique this stuff
  # dof_perm = _unique_sort_perm(dofs)
  # # el_perm = _unique_sort_perm(elements)

  # # do permutations
  # # TODO need to do this in different BC types
  # # since different BC types might want things
  # # organized differently
  # # blocks_new = blocks[el_perm]
  # dofs_new = dofs[dof_perm]
  # # elements_new = elements[el_perm]
  # nodes_new = nodes[dof_perm]
  # # sides_new = sides[el_perm]
  # # side_nodes_new = side_nodes[:, el_perm]

  # # resize!(blocks, length(blocks_new))
  # resize!(dofs, length(dofs_new))
  # # resize!(elements, length(elements_new))
  # resize!(nodes, length(nodes_new))
  # # resize!(sides, length(sides_new))
  # # resize!(side_nodes, size(side_nodes_new)...)
  # # reshape(side_nodes, size(side_nodes_new))

  # # copyto!(blocks, blocks_new)
  # copyto!(dofs, dofs_new)
  # # copyto!(elements, elements_new)
  # copyto!(nodes, nodes_new)
  # # copyto!(sides, sides_new)
  # # copyto!(side_nodes, side_nodes_new)

  return BCBookKeeping{typeof(side_nodes), typeof(blocks)}(
    blocks, dofs, elements, nodes, sides, side_nodes
  )
end

"""
$(TYPEDSIGNATURES)
"""
KA.get_backend(bk::BCBookKeeping) = KA.get_backend(bk.blocks)

function Base.show(io::IO, bk::BCBookKeeping)
  println(io, "Blocks                    = $(unique(bk.blocks))")
  println(io, "Number of active dofs     = $(length(bk.dofs))")
  println(io, "Number of active elements = $(length(bk.elements))")
  println(io, "Number of active nodes    = $(length(bk.nodes))")
  println(io, "Number of active sides    = $(length(bk.sides))")
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
abstract type AbstractBC{F <: Function} end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
abstract type AbstractBCContainer{
  B <: BCBookKeeping,
  T <: Union{<:Number, <:SVector},
  N,
  V <: AbstractArray{T, N}
} end

KA.get_backend(x::AbstractBCContainer) = KA.get_backend(x.vals)

function Base.show(io::IO, bc::AbstractBCContainer)
  println(io, "$(typeof(bc).name.name):")
  println(io, "$(bc.bookkeeping)")
end

"""
$(TYPEDSIGNATURES)
Wrapper that is generic for all architectures to
update bc values based on the stored function
"""
function update_bc_values!(bcs, funcs, X, t)
  for (bc, func) in zip(values(bcs), values(funcs))
    backend = KA.get_backend(bc)
    _update_bc_values!(bc, func, X, t, backend)
  end
  backend = KA.get_backend(X)
  KA.synchronize(backend)
  return nothing
end

include("DirichletBCs.jl")
include("NeumannBCs.jl")
