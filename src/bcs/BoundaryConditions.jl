struct VariableNameNotFoundError <: Exception
end

_var_not_found_err() = throw(VariableNameNotFoundError())

function _dof_index_from_var_name(dof, var_name)
  dof_index = 0
  found = false
  for name in names(dof.var)
      dof_index = dof_index + 1
      if var_name == name
          found = true
          break
      end
  end

  if dof_index == 0 || dof_index > length(names(dof.var))
    _var_not_found_err()
  end

  return dof_index
end

function _unique_sort_perm(array::AbstractArray{T, 1}) where T <: Number
  # chatgpt BS below. Make this not use a dict
  sorted_unique = sort(unique(array))
  id_map = Dict(x => findfirst(==(x), array) for x in sorted_unique)
  return [id_map[x] for x in sorted_unique]
end

const SetName = Union{Nothing, Symbol}

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
This struct is used to help with book keeping nodes, sides, etc.
for all types of boundary conditions.

TODO need to add a domain ID for extending to Schwarz
"""
struct BCBookKeeping{
  I <: Integer,
  V <: AbstractArray{I, 1},
  M <: AbstractArray{I, 2}
}
  blocks::V
  dofs::V
  elements::V
  nodes::V
  sides::V
  side_nodes::M
end

# TODO currently only works for H1 spaces
"""
$(TYPEDSIGNATURES)
"""
function BCBookKeeping(
  mesh, dof::DofManager, var_name::Symbol; #sset_name::Symbol
  block_name::SetName = nothing,
  nset_name::SetName = nothing,
  sset_name::SetName = nothing
)
  # check to ensure at least one name is supplied
  if block_name === nothing &&
     nset_name === nothing &&
     sset_name === nothing
    @assert false "Need to specify either a block, nodeset or sideset."
  end

  # now check to make sure only one name is not nothing
  if block_name !== nothing
    @assert nset_name === nothing && sset_name === nothing
  elseif nset_name !== nothing
    @assert block_name === nothing && sset_name === nothing
  elseif sset_name !== nothing  
    @assert block_name === nothing && nset_name === nothing
  end 

  # get dof index associated with this var
  dof_index = _dof_index_from_var_name(dof, var_name)

  # get sset specific fields
  all_dofs = reshape(1:length(dof), size(dof))

  if block_name !== nothing
    # for this case it is likely a DirichletBC or InitialCondition
    # so we really only need the nodes/dofs although this might
    # not be the case for say Hdiv or Hcurl fields...
    # TODO eventually set the blocks, could be useful maybe?
    blocks = Vector{Int64}(undef, 0)
    conns = getproperty(mesh.element_conns, block_name)
    nodes = sort(unique(conns.data))
    dofs = all_dofs[dof_index, nodes]
    elements = mesh.element_id_maps[block_name]
    # below 2 don't make sense for other mesh entity types
    sides = Vector{Int64}(undef, 0)
    side_nodes = Matrix{Int64}(undef, 0, 0)
  elseif nset_name !== nothing
    # for this case we only setup the "nodes" and "dofs"
    blocks = Vector{Int64}(undef, 0) # TODO we could eventually put the blocks present here
    nodes = mesh.nodeset_nodes[nset_name]
    dofs = all_dofs[dof_index, nodes]
    elements = Vector{Int64}(undef, 0) # TODO we could eventually put the elements present here
    # below 2 don't make sense for other mesh entity types
    sides = Vector{Int64}(undef, 0)
    side_nodes = Matrix{Int64}(undef, 0, 0)
  elseif sset_name !== nothing
    elements = mesh.sideset_elems[sset_name]
    nodes = mesh.sideset_nodes[sset_name]
    sides = mesh.sideset_sides[sset_name]
    side_nodes = mesh.sideset_side_nodes[sset_name]

    blocks = Vector{Int64}(undef, 0)

    # gather the blocks that are present in this sideset
    # TODO this isn't quite right
    for (n, val) in enumerate(values(mesh.element_id_maps))
      # note these are the local elem id to the block, e.g. starting from 1.
      indices_in_sset = indexin(val, elements)
      filter!(x -> x !== nothing, indices_in_sset)
      
      if length(indices_in_sset) > 0
        append!(blocks, repeat([n], length(indices_in_sset)))
      end
    end

    # setup dofs local to this BC
    # all_dofs = reshape(1:length(dof), size(dof))
    dofs = all_dofs[dof_index, nodes]
  else
    @assert false "Either you need to provide a nodeset, sideset or block"
  end

  return BCBookKeeping(
    blocks, dofs, elements, nodes, sides, side_nodes
  )
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
# abstract type AbstractBCContainer{
#   IT <: Integer,
#   VT <: Union{<:Number, <:SVector},
#   N,
#   IV <: AbstractArray{IT, 1},
#   IM <: AbstractArray{IT, 2},
#   VV <: AbstractArray{VT, N}
# } end
abstract type AbstractBCContainer end

KA.get_backend(x::AbstractBCContainer) = KA.get_backend(x.vals)

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
abstract type AbstractBCFunction{F} end

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
  return nothing
end

include("DirichletBCs.jl")
include("NeumannBCs.jl")
