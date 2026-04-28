struct EntityNameNotProvidedError <: AbstractFECError
  msg::String
end
_entity_not_provided_error(msg::String) = throw(EntityNameNotProvidedError(msg))
struct UnsupportedWeakBCError <: AbstractFECError
  msg::String
end
_unsupported_weak_bc_error(msg::String) = throw(UnsupportedWeakBCError(msg))
struct UnsureEntityTypeError <: AbstractFECError
  msg::String
end
_unsure_entity_type_error(msg::String) = throw(UnsureEntityTypeError(msg))
struct VariableNameNotFoundError <: Exception
end
_var_not_found_err() = throw(VariableNameNotFoundError())

function _dof_index_from_var_name(dof, var_name)
  # dbc/nbc or robin for scalar case
  dof_index = findfirst(x -> x == var_name, names(dof.var))
  if dof_index === nothing
    # try a vector/tensor neumann/robin bc type
    dof_index = findfirst(x -> occursin("$(var_name)_", x), names(dof.var))

    # if we still haven't found anything, this variable likely doesn't exist
    if dof_index === nothing
      _var_not_found_err()
    end
  end
  return dof_index
end

function _unique_sort_perm(array::AbstractArray{T, 1}) where T <: Number
  sorted_unique = sort(unique(array))
  id_map = Dict{T, Int}(x => findfirst(==(x), array) for x in sorted_unique)
  return [id_map[x] for x in sorted_unique]
end

const SetName = Union{Nothing, String}

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
  mesh, dof::DofManager, var_name::String;
  block_name::SetName = nothing,
  nset_name::SetName = nothing,
  sideset_name::SetName = nothing
)
  if block_name === nothing && nset_name === nothing && sideset_name === nothing
    _entity_not_provided_error(
      "block_name, nset_name, or sideset_name required" *
      " as input arguments in DirichletBC"
    )
  end
  count = (block_name !== nothing) +
          (nset_name !== nothing) +
          (sideset_name !== nothing)
  if count != 1
    _unsure_entity_type_error("More than one entity type specificed in DirichletBC")
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
  elseif sideset_name !== nothing
    elements = mesh.sideset_elems[sideset_name]
    nodes = mesh.sideset_nodes[sideset_name]
    sides = mesh.sideset_sides[sideset_name]
    side_nodes = mesh.sideset_side_nodes[sideset_name]

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
  RV <: AbstractArray{<:Union{<:Number, <:SVector}, 2}
} <: AbstractBCContainer{IV, RV} end

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
abstract type AbstractBCs{Funcs} end

# required interface to implement
"""
$(TYPEDSIGNATURES)
Required interface to implement. Will typically take the
form of

``
function update_bc_values!(bcs::SomeBCs, X, t, args...)
...
end
``
"""
function update_bc_values! end

Base.length(bcs::AbstractBCs) = length(bcs.bc_funcs)

# function Base.show(io::IO, bcs::AbstractBCs)
#   type = typeof(bcs).name.name
#   for (n, (cache, func)) in enumerate(zip(bcs.bc_caches, bcs.bc_funcs))
#     show(io, "$(type)_$n")
#     show(io, cache)
#     show(io, func)
#     show(io, "\n")
#   end
# end

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
  bks = map((v, s) -> BCBookKeeping(mesh, dof, v; sideset_name = s), vars, sets)

  fspace = function_space(dof)
  new_bcs = type[]
  new_funcs = Function[]
  block_ids = Int[]
  block_names = String[]
  # sideset_ids = Int[]
  # sideset_names = String[]
  for (bk, func) in zip(bks, funcs)
    blocks = sort(unique(bk.blocks))

    # TODO fix this
    if length(blocks) > 1
      @error "Neumann BCs present on multiple blocks will likely fail"
    end

    for block in blocks
      block_name = mesh.element_block_names_map[block]
      block_id = findfirst(x -> x == block_name, mesh.element_block_names)
      push!(block_ids, block_id)
      push!(block_names, block_name)

      ids = findall(x -> x == block, bk.blocks)
      new_blocks = bk.blocks[ids]
      new_elements = bk.elements[ids]
      new_sides = bk.sides[ids]
      new_side_nodes = bk.side_nodes[:, ids]

      # TODO update nodes and dofs
      new_bk = BCBookKeeping(new_blocks, bk.dofs, new_elements, bk.nodes, new_sides, new_side_nodes)
      id = findfirst(x -> x == block_name, mesh.element_block_names)
      ref_fe = fspace.ref_fes[id]
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
          conns, new_bk.elements, new_bk.sides, vals
        )
      elseif type <: RobinBCContainer
        # dvalsdu = copy(vals)
        dvalsdu = zeros(SMatrix{ND, ND, Float64, ND * ND}, NQ, length(bk.sides))
        new_bc = type(
          conns, new_bk.elements, new_bk.sides, vals, dvalsdu
        )
      else
        _unsupported_weak_bc_error("Unsupported weak bc type $type encountered in _setup_weakly_enforced_bc_container")
      end
      push!(new_bcs, new_bc)
      push!(new_funcs, func)
    end
  end
  return new_bcs, new_funcs, block_ids, block_names
end

function _setup_sideset(mesh, dof, bc)
  # just check that this is here..., if its not
  # below method call will throw an error
  _dof_index_from_var_name(dof, bc.var_name)

  # get sset specific fields
  sideset_name = bc.sset_name
  elements = mesh.sideset_elems[sideset_name]
  sides = mesh.sideset_sides[sideset_name]
  side_nodes = mesh.sideset_side_nodes[sideset_name]

  # gather the blocks that are present in this sideset
  # and also map global element id to local element id
  blocks = Vector{Int64}(undef, 0)
  for (n, val) in enumerate(values(mesh.element_id_maps))
    # note these are the local elem id to the block, e.g. starting from 1.
    indices_in_sset = indexin(val, elements)
    filter!(x -> x !== nothing, indices_in_sset)

    if length(indices_in_sset) > 0
      append!(blocks, repeat([n], length(indices_in_sset)))
    end
  end

  unique_block_ids = sort(unique(blocks))
  @assert length(unique_block_ids) == 1 "Sidesets need to be in a single block"
  block_name = mesh.element_block_names[unique_block_ids[1]]
  ids = findall(x -> x == unique_block_ids[1], blocks)
  indices_in_sset = indexin(elements, mesh.element_id_maps[block_name])
  filter!(x -> x !== nothing, indices_in_sset)
  elements = convert(Vector{Int}, indices_in_sset)

  # end bc bookkeeping

  block_id = findfirst(x -> x == block_name, mesh.element_block_names)

  # this probably doesn't do anything for when we just have one block
  blocks = blocks[ids]
  elements = elements[ids]
  sides = sides[ids]
  side_nodes = side_nodes[ids]

  # setup cache and connectivity
  fspace = function_space(dof)
  # NQs = Int[]
  # # foreach_block(fspace.block_to_ref_fe_id) do ref_fe, b
  # foreach_block(fspace) do ref_fe, b
  #   if b == block_id
  #     # NQ = num_surface_quadrature_points(ref_fe)::Int
  #     # NQ = 1
  #     # push!(NQs, NQ)
  #     push!(NQs, NQ)
  #   end
  # end
  # @assert length(NQs) == 1
  # NQ = NQs[1]
  # @assert NQ != -1 "Bad quadrature"
  nqs = -1 * ones(Int, MAX_BLOCKS)
  foreach_block(fspace) do ref_fe, b
    if b == block_id
      nq = num_surface_quadrature_points(ref_fe)
      nqs[b] = nq
    end
  end
  index = findfirst(x -> x != -1, nqs)
  NQ = nqs[index]
  ND = size(dof, 1)

  # need to carefully check this guy
  conns = mesh.element_conns[block_name][:, elements]
  conns = Connectivity(Matrix{Int}[conns])
  vals = zeros(SVector{ND, Float64}, NQ, length(sides))

  return block_id, block_name, conns, elements, sides, vals
end
