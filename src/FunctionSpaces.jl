const _default_p = Dict{String, Int}(
  "HEX"     => 1,
  "HEX8"    => 1,
  "QUAD"    => 1,
  "QUAD4"   => 1,
  "QUAD9"   => 2,
  "TRI"     => 1,
  "TRI3"    => 1,
  "TRI6"    => 2,
  "TET"     => 1,
  "TETRA"   => 1,
  "TETRA4"  => 1,
  "TETRA10" => 2
)
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

function _setup_ref_fes(mesh::AbstractMesh, interp_type, p_degree = nothing, q_degree = nothing)
  block_names = mesh.element_block_names
  ref_fes = ReferenceFE[]
  for elem_name in values(mesh.element_types)
    elem_type = elem_type_map[uppercase(String(elem_name))]
    if p_degree === nothing
      p_degree = _default_p[uppercase(String(elem_name))]
    end

    if q_degree === nothing
      q_degree = _default_q[uppercase(String(elem_name))]
    end
    ref_fe = ReferenceFE(elem_type{interp_type, p_degree}(), GaussLobattoLegendre(q_degree))
    push!(ref_fes, ref_fe)
  end
  ref_fes = NamedTuple{tuple(values(block_names)...)}(tuple(ref_fes...))
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
  elem_id_maps::Vector{Vector{IT}} # TODO create new type for ID map similar to connectivity
  ref_fes::RefFEs
end

function FunctionSpace(
  mesh::AbstractMesh, ::Type{H1Field}, interp_type; 
  p_degree::Union{Int, Nothing} = nothing,
  q_degree::Union{Int, Nothing} = nothing
)
  # TODO move to some common function so we can use it across
  # all constructors
  if p_degree !== nothing
    # some error checking on p_degree input
    if p_degree < 0
      @assert false "Bad polynomial degree $(p_degree)"
    elseif p_degree == 0
      @assert false "TODO 0 order elements"
    else
      ref_fes = _setup_ref_fes(mesh, interp_type, p_degree, q_degree)
      coords, conns = create_higher_order_mesh(mesh, H1Field, interp_type, p_degree)
      conns = Connectivity([val for val in values(conns)])
    end
  else
    ref_fes = _setup_ref_fes(mesh, interp_type, q_degree)
    coords = mesh.nodal_coords
    conns = Connectivity([val for val in values(mesh.element_conns)])
  end
  # conns = Connectivity([mesh.element_conns[name] for name in mesh.element_block_names])
  # conns = Connectivity([val for val in values(mesh.element_conns)])
  # elem_id_maps = [mesh.element_id_maps[name] for name in mesh.element_block_names]
  elem_id_maps = [val for val in values(mesh.element_id_maps)]
  return FunctionSpace(coords, conns, elem_id_maps, ref_fes)
end

function FunctionSpace(
  mesh::AbstractMesh, ::Type{HdivField}, ::Type{Lagrange},
  p_degree::Union{Int, Nothing} = nothing,
  q_degree::Union{Int, Nothing} = nothing
)
  if p_degree !== nothing
    @assert p_degree == 0 "Only 0 degree Hdiv elements supported so far"
    # TODO finish
  else

  end
end

function FunctionSpace(
  mesh::AbstractMesh, ::Type{L2Field}, ::Type{Lagrange};
  q_degree = nothing
)
  ref_fes = _setup_ref_fes(mesh, Lagrange, q_degree)

  # conns = Connectivity([mesh.element_conns[name] for name in mesh.element_block_names])
  conns = Connectivity([val for val in values(mesh.element_conns)])
  coords = L2Field(map(x -> mesh.nodal_coords[:, x], [values(mesh.element_conns)...]))

  new_conns = Array{Int, 2}[]
  offset = 1
  # for name in mesh.element_block_names
  for name in keys(mesh.element_conns)
    conn = mesh.element_conns[name]
    push!(new_conns, reshape(offset:offset + length(conn) - 1, size(conn)...))
    offset += size(conn, 1) * size(conn, 2)
  end
  conns = Connectivity(new_conns)

  return FunctionSpace(coords, conns, Vector{Vector{Int}}(undef, 0), ref_fes)
end

function Adapt.adapt_structure(to, fspace::FunctionSpace)
  coords = adapt(to, fspace.coords)
  elem_conns = adapt(to, fspace.elem_conns)
  elem_id_maps = fspace.elem_id_maps
  ref_fes = adapt(to, fspace.ref_fes)
  return FunctionSpace(coords, elem_conns, elem_id_maps, ref_fes)
end

function Base.show(io::IO, fspace::FunctionSpace)
  println(io, "FunctionSpace:")
  println(io, "  Type: $(typeof(fspace.coords).name.name)")
  for (key, ref_fe) in enumerate(fspace.ref_fes)
    println(io, "    Block: $key")
  end
end

function block_size(fspace::FunctionSpace, b::Int)
  return (num_entities_per_element(fspace, b), num_elements(fspace, b))
end

function connectivity(fspace::FunctionSpace)
  return fspace.elem_conns
end

function connectivity(fspace::FunctionSpace, b::Int)
  return connectivity(fspace.elem_conns, b)
end

function coordinates(fspace::FunctionSpace)
  return fspace.coords
end

function num_blocks(fspace::FunctionSpace)
  return num_blocks(fspace.elem_conns)
end

function num_elements(fspace::FunctionSpace, b::Int)
  return num_elements(fspace.elem_conns, b)
end

function num_entities_per_element(fspace::FunctionSpace, b::Int)
  return num_entities_per_element(fspace.elem_conns, b)
end

function unsafe_connectivity(fspace::FunctionSpace, e::Int, b::Int)
  return unsafe_connectivity(fspace.elem_conns, e, b)
end

# # this is an H1 method, need to specialize
# function _create_linear_edges(fspace::FunctionSpace, ::Type{<:H1Field}, ::Type{<:Lagrange})
#   # need to assert this is already a linear function space
#   for re in fspace.ref_fes
#     if dimension(re) == 2
#       @assert polynomial_degree(re) == 1
#     else
#       @assert false "Unsupported dimension = $(dimension(re))"
#     end
#   end

#   # maps canonical edges
#   # to a vector of triplets where each element of that 
#   # vector corresponds to (el_id, local_edge_num, orientation)
#   edge2elem = Dict{NTuple{2, Int}, Vector{NTuple{3, Int}}}()

#   for b in 1:num_blocks(fspace)
#     conn = connectivity(fspace, b)
#     el_ids = fspace.elem_id_maps[b]
#     re = values(fspace.ref_fes)[b]
#     for e in axes(conn, 2)
#       local_edges = _create_local_edges(conn, re, e)
#       el_id = el_ids[e]
#       for (le_num, le) in enumerate(local_edges)
#         ce = _canonical_edge(le...)
#         orientation = le[1] < le[2] ? 1 : -1
#         push!(
#           get!(edge2elem, ce, Vector{NTuple{3, Int}}()),
#           (el_id, le_num, orientation)
#         )
#       end
#     end
#   end
#   return edge2elem
# end

