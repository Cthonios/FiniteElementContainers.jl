# TODO remove these H1, etc. types since now fields
# are appropriately called these
# 
# we can read the fspace type off of coords if
# we appropriately make that coords for e.g. H1, L2, etc.
#
abstract type AbstractFunctionSpaceType end

struct H1 <: AbstractFunctionSpaceType
end

struct Hcurl <: AbstractFunctionSpaceType
end

struct Hdiv <: AbstractFunctionSpaceType
end

struct L2Element <: AbstractFunctionSpaceType
end

struct L2Quadrature <: AbstractFunctionSpaceType
end

abstract type AbstractFunctionSpace end

struct FunctionSpace{
  Coords, 
  ElemConns, ElemIdMaps, 
  FSpaceType,
  RefFEs,
  SSetElems, SSetNodes, SSetSides
} <: AbstractFunctionSpace
  coords::Coords
  elem_conns::ElemConns
  elem_id_maps::ElemIdMaps
  fspace_type::FSpaceType
  ref_fes::RefFEs
  sideset_elems::SSetElems
  sideset_nodes::SSetNodes
  sideset_sides::SSetSides
end

function FunctionSpace(mesh::AbstractMesh, ::Type{H1}, interp_type, q_degree)
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

  return FunctionSpace(
    coords, 
    mesh.element_conns, mesh.element_id_maps, 
    H1(), 
    ref_fes,
    mesh.sideset_elems, mesh.sideset_nodes, mesh.sideset_sides
  )
end

function FunctionSpace(mesh::AbstractMesh, ::Type{L2Element}, interp_type)
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
  println(io, "  Type: $(fspace.fspace_type)")
  for (key, ref_fe) in enumerate(fspace.ref_fes)
    println(io, "    Block: $key")
    # println(io, "      Number of elements: $(fspace)")
    # println(io, "  $ref_fe")
  end
end

function connectivity(fspace, block)
  return Base.getproperty(fspace.elem_conns, block)
end

function _connectivity(fspace, e::Int, block)
  temp = @views Base.getproperty(fspace.elem_conns, block)[:, e]
  # NN = num_vertices(fspace.ref_fes[block])
  # return SVector{NN, eltype(temp)}(temp)
  return temp
end

connectivity(fspace, e, block::Symbol) = _connectivity(fspace, e, Val(block))

function connectivity(fspace, n::Int, e::Int, block)
  return @views Base.getproperty(fspace.elem_conns, block)[n, e]
end 

coordinates(fspace::FunctionSpace) = fspace.coords

function coordinates(fspace::FunctionSpace, e, block)
  conn = _connectivity(fspace, e, block)
  return @views fspace.coords[:, conn]
end
