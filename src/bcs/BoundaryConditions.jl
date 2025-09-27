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
  mesh, dof::DofManager, var_name::Symbol, sset_name::Symbol
)
  # get dof index associated with this var
  dof_index = _dof_index_from_var_name(dof, var_name)

  fspace = function_space(dof)

  # get sset specific fields
  elements = getproperty(mesh.sideset_elems, sset_name)
  nodes = getproperty(mesh.sideset_nodes, sset_name)
  sides = getproperty(mesh.sideset_sides, sset_name)
  side_nodes = getproperty(mesh.sideset_side_nodes, sset_name)

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
  all_dofs = reshape(1:length(dof), size(dof))
  dofs = all_dofs[dof_index, nodes]

  return BCBookKeeping(
    blocks, dofs, elements, nodes, sides, side_nodes
  )
end

function BCBookKeeping(mesh, sset_name::Symbol)
    # get sset specific fields
    elements = getproperty(mesh.sideset_elems, sset_name)
    nodes = getproperty(mesh.sideset_nodes, sset_name)
    sides = getproperty(mesh.sideset_sides, sset_name)
    side_nodes = getproperty(mesh.sideset_side_nodes, sset_name)
  
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

    dofs = Vector{Int64}(undef, 0)
      return BCBookKeeping(
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
  IT <: Integer,
  VT <: Union{<:Number, <:SVector},
  N,
  IV <: AbstractArray{IT, 1},
  IM <: AbstractArray{IT, 2},
  VV <: AbstractArray{VT, N}
} end

KA.get_backend(x::AbstractBCContainer) = KA.get_backend(x.vals)

function Base.show(io::IO, bc::AbstractBCContainer)
  println(io, "$(typeof(bc).name.name):")
  println(io, "$(bc.bookkeeping)")
end

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
