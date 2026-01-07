# const default_p = Dict{String, Int}(

# )
const _default_q = Dict{String, Int}(
  "HEX"     => 2,
  "HEX8"    => 2,
  "QUAD"    => 2,
  "QUAD4"   => 2,
  "QUAD9"   => 2,
  "TRI"     => 2,
  "TRI3"    => 2,
  "TRI6"    => 2,
  "TET"     => 2,
  "TETRA"   => 2,
  "TETRA4"  => 2,
  "TETRA10" => 2
)

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
  IT <: Integer,
  IV <: AbstractVector{IT},
  Coords,
  RefFEs
} <: AbstractFunctionSpace
  coords::Coords
  elem_conns::Connectivity{IT, IV}
  ref_fes::RefFEs
end

function _setup_ref_fes(mesh::AbstractMesh, interp_type, q_degree = nothing)
  block_names = mesh.element_block_names
  ref_fes = ReferenceFE[]
  for elem_name in mesh.element_types
    elem_type = elem_type_map[uppercase(String(elem_name))]
    if q_degree === nothing
      q_degree = _default_q[uppercase(String(elem_name))]
    end
    ref_fe = ReferenceFE(elem_type{interp_type, q_degree}())
    push!(ref_fes, ref_fe)
  end
  ref_fes = NamedTuple{tuple(block_names...)}(tuple(ref_fes...))
  return ref_fes
end

function _setup_quad_coords(mesh, X, conns, ref_fe) 
  NE = size(conns, 2)
  NNPE = size(conns, 1)
  NQ = num_quadrature_points(ref_fe)
  ND = num_fields(mesh.nodal_coords)
  coords_temp = zeros(ND, NQ, NE)

  for e in axes(X, 3)
    X_el = SMatrix{ND, NNPE, Float64, ND * NNPE}(@views X[:, :, e])
    for q in 1:NQ
      X_q = X_el * shape_function_value(ref_fe, q)
      coords_temp[:, q, e] .= X_q
    end
  end
  return coords_temp
end

function FunctionSpace(
  mesh::AbstractMesh, ::Type{H1Field}, ::Type{Lagrange}; 
  q_degree = nothing
)
  ref_fes = _setup_ref_fes(mesh, Lagrange, q_degree)

  conns = [values(mesh.element_conns)...]
  conns = Connectivity(conns)

  return FunctionSpace(mesh.nodal_coords, conns, ref_fes)
end

function FunctionSpace(
  mesh::AbstractMesh, ::Type{L2Field}, ::Type{Lagrange};
  q_degree = nothing
)
  ref_fes = _setup_ref_fes(mesh, Lagrange, q_degree)

  conns = [values(mesh.element_conns)...]
  coords = L2Field(map(x -> mesh.nodal_coords[:, x], [values(mesh.element_conns)...]))

  new_conns = Array{Int, 2}[]
  offset = 1
  for conn in conns
    push!(new_conns, reshape(offset:offset + length(conn) - 1, size(conn)...))
    offset += size(conn, 1) * size(conn, 2)
  end
  conns = Connectivity(new_conns)

  return FunctionSpace(coords, conns, ref_fes)
end

function Adapt.adapt_structure(to, fspace::FunctionSpace)
  coords = adapt(to, fspace.coords)
  elem_conns = adapt(to, fspace.elem_conns)
  ref_fes = adapt(to, fspace.ref_fes)
  return FunctionSpace(coords, elem_conns, ref_fes)
end

function Base.show(io::IO, fspace::FunctionSpace)
  println(io, "FunctionSpace:")
  println(io, "  Type: $(typeof(fspace.coords).name.name)")
  for (key, ref_fe) in enumerate(fspace.ref_fes)
    println(io, "    Block: $key")
  end
end

function connectivity(fspace::FunctionSpace, b::Int)
  return connectivity(fspace.elem_conns, b)
end

function num_blocks(fspace::FunctionSpace)
  return num_blocks(fspace.elem_conns)
end

function num_elements(fspace::FunctionSpace, b::Int)
  return num_elements(fspace.elem_conns, b)
end
