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
    conns = mesh.element_conns[block_name]
    nodes = sort(unique(conns))
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
    # and also map global element id to local element id
    # TODO this isn't quite right
    for (n, val) in enumerate(values(mesh.element_id_maps))
      # note these are the local elem id to the block, e.g. starting from 1.
      indices_in_sset = indexin(val, elements)
      filter!(x -> x !== nothing, indices_in_sset)

      if length(indices_in_sset) > 0
        append!(blocks, repeat([n], length(indices_in_sset)))
      end
    end

    @assert length(unique(blocks)) == 1 "Sidesets need to be in a single block"
    block_name = mesh.element_block_names[blocks[1]]
    indices_in_sset = indexin(elements, mesh.element_id_maps[block_name])
    filter!(x -> x !== nothing, indices_in_sset)
    elements = convert(Vector{Int}, indices_in_sset)
    # display(elements)

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
abstract type AbstractBCContainer{
  IV <: AbstractArray{<:Integer, 1},
  RV <: AbstractArray
} end

KA.get_backend(x::AbstractBCContainer) = KA.get_backend(x.vals)

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
abstract type AbstractWeaklyEnforcedBCContainer{
  IT <: Integer,
  IV <: AbstractArray{IT, 1},
  RV <: AbstractArray{<:Union{<:Number, <:SVector}, 2},
  RE <: ReferenceFE
} <: AbstractBCContainer{IV, RV} end

function Adapt.adapt_structure(to, bc::AbstractWeaklyEnforcedBCContainer)
  el_conns = adapt(to, bc.element_conns)
  elements = adapt(to, bc.elements)
  sides = adapt(to, bc.sides)
  ref_fe = adapt(to, bc.ref_fe)
  vals = adapt(to, bc.vals)
  type = eval(typeof(bc).name.name)
  return type(el_conns, elements, sides, ref_fe, vals)
end

Base.length(bc::AbstractWeaklyEnforcedBCContainer) = size(bc.vals, 2)

function Base.show(io::IO, bc::AbstractWeaklyEnforcedBCContainer)
  println(io, "$(typeof(bc).name.name):")
  # println(io, "Blocks                    = $(unique(bk.blocks))")
  println(io, "  Number of active elements = $(length(bc.elements))")
  # println(io, "  Number of active nodes    = $(length(bc.side_nodes))")
  println(io, "  Number of active sides    = $(length(bc.sides))")
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
abstract type AbstractBCFunction{F} end

function Base.show(io::IO, func::AbstractBCFunction)
  println(io, typeof(func.func).name.name)
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
abstract type AbstractBCs{
  Funcs <: NamedTuple
} end

function Adapt.adapt_structure(to, bcs::AbstractBCs)
  type = typeof(bcs).name.name
  return eval(type)(
    map(x -> adapt(to, x), bcs.bc_caches),
    adapt(to, bcs.bc_funcs)
  )
end

Base.length(bcs::AbstractBCs) = length(bcs.bc_funcs)

function Base.show(io::IO, bcs::AbstractBCs)
  type = typeof(bcs).name.name
  for (n, (cache, func)) in enumerate(zip(bcs.bc_caches, bcs.bc_funcs))
    show(io, "$(type)_$n")
    show(io, cache)
    show(io, func)
    show(io, "\n")
  end
end

"""
$(TYPEDSIGNATURES)
Wrapper that is generic for all architectures to
update bc values based on the stored function
"""
function update_bc_values!(bcs::AbstractBCs, X, t, args...)
  for (bc, func) in zip(values(bcs.bc_caches), values(bcs.bc_funcs))
    _update_bc_values!(bc, func, X, t, args...)
  end
  return nothing
end

include("DirichletBCs.jl")
include("NeumannBCs.jl")
include("PeriodicBCs.jl")
include("RobinBCs.jl")
include("Sources.jl")

# returns vectors
# callers are responsible for converting to named tuples
# or other datatypes
function _setup_weakly_enforced_bc_container(mesh, dof, bcs, type)
  sets = map(x -> x.sset_name, bcs)
  vars = map(x -> x.var_name, bcs)
  funcs = map(x -> x.func, bcs)
  # NOTE neumann bcs must be present on a sideset
  # so that is the only mesh entity that will be
  # supported for this BC type
  bks = map((v, s) -> BCBookKeeping(mesh, dof, v; sset_name = s), vars, sets)

  fspace = function_space(dof)
  new_bcs = type[]
  new_funcs = Function[]
  for (bk, func) in zip(bks, funcs)
    blocks = sort(unique(bk.blocks))

    # TODO fix this
    if length(blocks) > 1
      @error "Neumann BCs present on multiple blocks will likely fail"
    end

    for block in blocks
      block_name = mesh.element_block_names[block]
      ids = findall(x -> x == block, bk.blocks)
      new_blocks = bk.blocks[ids]
      new_elements = bk.elements[ids]
      new_sides = bk.sides[ids]
      new_side_nodes = bk.side_nodes[:, ids]

      # TODO update nodes and dofs
      new_bk = BCBookKeeping(new_blocks, bk.dofs, new_elements, bk.nodes, new_sides, new_side_nodes)
      ref_fe = getproperty(fspace.ref_fes, block_name)
      NQ = num_surface_quadrature_points(ref_fe)
      ND = length(dof.var)

      # TODO below isn't correct
      # we need to map bk.elements using the block element id map
      # NOTE I think that is currently handled in BCBookKeeping
      # by setting bk.elements to the indices where the side set
      # elements live in the block element id map
      conns = mesh.element_conns[block_name][:, bk.elements]

      conns = Connectivity([conns])
      vals = zeros(SVector{ND, Float64}, NQ, length(bk.sides))

      if type <: NeumannBCContainer
        new_bc = type(
          conns, new_bk.elements, new_bk.sides, ref_fe, vals
        )
      elseif type <: RobinBCContainer
        # dvalsdu = copy(vals)
        dvalsdu = zeros(SMatrix{ND, ND, Float64, ND * ND}, NQ, length(bk.sides))
        new_bc = type(
          conns, new_bk.elements, new_bk.sides, ref_fe, vals, dvalsdu
        )
      else
        @assert false "Unsupported bc type"
      end
      push!(new_bcs, new_bc)
      push!(new_funcs, func)
    end
  end
  return new_bcs, new_funcs
end
