
struct FunctionSpace{Coords, ElemConns, ElemIdMaps, RefFEs}
  coords::Coords
  elem_conns::ElemConns
  elem_id_maps::ElemIdMaps
  ref_fes::RefFEs
end

function FunctionSpace(mesh::AbstractMesh, interp_type; q_degree=2)
  # block_names = Symbol.(valkeys(mesh.element_conns))
  block_names = mesh.element_block_names
  ref_fes = []
  for elem_name in mesh.element_types
    elem_type = elem_type_map[elem_name]
    # TODO hardcoded q degree right now
    ref_fe = ReferenceFE(elem_type{interp_type, q_degree}())
    push!(ref_fes, ref_fe)
  end
  ref_fes = NamedTuple{tuple(block_names...)}(tuple(ref_fes...))

  if interp_type == Lagrange
    coords = mesh.nodal_coords
  else
    throw(ErrorException("Unssuported interp_type $interp_type"))
  end

  return FunctionSpace(coords, mesh.element_conns, mesh.element_id_maps, ref_fes)
end

function Base.show(io::IO, fspace::FunctionSpace)
  println(io, "FunctionSpace:")
  for (key, ref_fe) in enumerate(fspace.ref_fes)
    println(io, "  Block: $key")
    println(io, "  $ref_fe")
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
