"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
abstract type AbstractFunctionSpace end

# Need to add dof conns back in.
"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
struct FunctionSpace{
  Coords <: AbstractField, 
  ElemConns, ElemIdMaps, 
  # FSpaceType,
  RefFEs,
  SSetElems, SSetNodes, SSetSides
} <: AbstractFunctionSpace
  coords::Coords
  elem_conns::ElemConns
  elem_id_maps::ElemIdMaps
  # fspace_type::FSpaceType
  ref_fes::RefFEs
  sideset_elems::SSetElems
  sideset_nodes::SSetNodes
  sideset_sides::SSetSides
end

function FunctionSpace(mesh::AbstractMesh, ::Type{H1Field}, interp_type, q_degree)
  # block_names = Symbol.(valkeys(mesh.element_conns))
  block_names = mesh.element_block_names
  ref_fes = ReferenceFE[]
  for elem_name in mesh.element_types
    elem_type = elem_type_map[elem_name]
    ref_fe = ReferenceFE(elem_type{interp_type, q_degree}())
    push!(ref_fes, ref_fe)
  end
  ref_fes = NamedTuple{tuple(block_names...)}(tuple(ref_fes...))

  if interp_type == Lagrange
    coords = mesh.nodal_coords
  else
    throw(ErrorException("Unssuported interp_type $interp_type"))
  end

  # create dof conns arrays

  return FunctionSpace(
    coords, 
    mesh.element_conns, mesh.element_id_maps, 
    # H1(), 
    ref_fes,
    mesh.sideset_elems, mesh.sideset_nodes, mesh.sideset_sides
  )
end

function FunctionSpace(mesh::AbstractMesh, ::Type{L2ElementField}, interp_type)
  block_names = mesh.element_block_names
  ref_fes = ReferenceFE[]
  for elem_name in mesh.element_types
    elem_type = elem_type_map[elem_name]
    ref_fe = ReferenceFE(elem_type{interp_type, 1}())
    push!(ref_fes, ref_fe)
  end
  ref_fes = NamedTuple{tuple(block_names...)}(tuple(ref_fes...))

  # TODO need to map nodal coordinates from mesh to element centroids
  @assert false "Unfinished method"

end

function FunctionSpace(mesh::AbstractMesh, space_type, interp_type; q_degree=2)
  return FunctionSpace(mesh, space_type, interp_type, q_degree)
end

function Base.show(io::IO, fspace::FunctionSpace)
  println(io, "FunctionSpace:")
  println(io, "  Type: $(typeof(fspace.coords).name.name)")
  for (key, ref_fe) in enumerate(fspace.ref_fes)
    println(io, "    Block: $key")
    # println(io, "      Number of elements: $(fspace)")
    # println(io, "  $ref_fe")
  end
end

function connectivity(fspace, block)
  return Base.getproperty(fspace.elem_conns, block)
end

function connectivity(fspace, e::Int, block::Symbol)
  temp = @views Base.getproperty(fspace.elem_conns, block)[:, e]
  # NN = num_vertices(fspace.ref_fes[block])
  # return SVector{NN, eltype(temp)}(temp)
  return temp
end

function connectivity(fspace, e::Int, block_num::Int)
  block_sym = keys(fspace.elem_conns)[block_num]
  return @views Base.getproperty(fspace.elem_conns, block_sym)[:, e]
end

# connectivity(fspace, e, block::Symbol) = _connectivity(fspace, e, Val(block))

function connectivity(fspace, n::Int, e::Int, block)
  return @views Base.getproperty(fspace.elem_conns, block)[n, e]
end 

coordinates(fspace::FunctionSpace) = fspace.coords

# function coordinates(fspace::FunctionSpace, e, block)
#   conn = _connectivity(fspace, e, block)
#   return @views fspace.coords[:, conn]
# end

function dof_connectivity(fspace, e, block, n_dofs)
  ids = reshape(1:length(fspace.coords), size(fspace.coords)...)
  block_name = keys(fspace.elem_conns)[block]
  conns = getproperty(fspace.elem_conns, block_name)
  # dof_conn = @views reshape(ids[:, conns[:, e]], n_dofs * size(conns, 1))
  dof_conn = reshape(ids[:, conns[:, e]], n_dofs * size(conns, 1))
  return dof_conn
end

function map_shape_function_gradients(X, ∇N_ξ)
  J     = (X * ∇N_ξ)'
  J_inv = inv(J)
  ∇N_X  = (J_inv * ∇N_ξ')'
  return ∇N_X
end

function num_dimensions(fspace::FunctionSpace, block_sym)
  return ReferenceFiniteElements.dimension(getproperty(fspace.ref_fes, block_sym))
end

function num_elements(fspace::FunctionSpace, block_sym)
  return num_elements(getproperty(fspace.elem_conns, block_sym))
end

function num_nodes_per_element(fspace::FunctionSpace, block_sym)
  return ReferenceFiniteElements.num_vertices(getproperty(fspace.ref_fes, block_sym))
end 

function num_q_points(fspace::FunctionSpace, block_sym)
  return ReferenceFiniteElements.num_quadrature_points(getproperty(fspace.ref_fes, block_sym))
end
