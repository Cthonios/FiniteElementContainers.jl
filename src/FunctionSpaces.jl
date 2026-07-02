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
  "TET4"    => 2,
  "TETRA"   => 2,
  "TETRA4"  => 2,
  "TETRA10" => 2
)
const _el_name_to_juliac_safe_id = Dict{String, Int}(
  "HEX"     => 1,
  "HEX8"    => 1,
  "QUAD"    => 2,
  "QUAD4"   => 2,
  "QUAD9"   => 3,
  "TRI"     => 4,
  "TRI3"    => 4,
  "TRI6"    => 5,
  "TET"     => 6,
  "TET4"    => 6,
  "TETRA"   => 6,
  "TETRA4"  => 6,
  "TETRA10" => 7
)
# Lagrange elements
# defaulting to fully integrated elements for now
const _juliac_safe_ref_fes = (
  ReferenceFE(Hex{Lagrange, 1}(), GaussLobattoLegendre(2, 2)),  # HEX8 for Lagrange
  ReferenceFE(Quad{Lagrange, 1}(), GaussLobattoLegendre(2, 2)), # QUAD4 for Lagrange
  ReferenceFE(Quad{Lagrange, 2}(), GaussLobattoLegendre(2, 2)), # QUAD9 for Lagrange
  ReferenceFE(Tri{Lagrange, 1}(), GaussLobattoLegendre(2, 2)),  # Tri3 for Lagrange
  ReferenceFE(Tri{Lagrange, 2}(), GaussLobattoLegendre(2, 2)),  # Tri3 for Lagrange
  ReferenceFE(Tet{Lagrange, 1}(), GaussLobattoLegendre(2, 2)),  # Tri3 for Lagrange
  ReferenceFE(Tet{Lagrange, 2}(), GaussLobattoLegendre(2, 2))   # Tri3 for Lagrange
)
const MAX_BLOCKS = 16

function _setup_juliac_safe_block_to_ref_fe_id(mesh::AbstractMesh)
  names = mesh.element_block_names
  el_types = map(x -> _el_name_to_juliac_safe_id[mesh.element_types[x]], names)
  N = length(names)
  return ntuple(i -> i <= N ? el_types[i] : -1, Val(MAX_BLOCKS))  # replace `i` with your actual block value
end

function _setup_block_to_ref_fe_id(mesh::AbstractMesh)
  return 1:length(mesh.element_types) |> collect
end

function _setup_block_to_ref_fe_id(mesh::AbstractMesh, ::Val{is_juliac_safe}) where is_juliac_safe
  if is_juliac_safe
    return _setup_juliac_safe_block_to_ref_fe_id(mesh)
  else
    return _setup_block_to_ref_fe_id(mesh)
  end
end

"""
default code path that sets up ref fes as a namedtuple
"""
function _setup_ref_fes(
  mesh::AbstractMesh, 
  interp_type, p_degree,
  q_type::Type{<:ReferenceFiniteElements.AbstractQuadratureType}, q_degree
)
  block_names = mesh.element_block_names
  ref_fes = ReferenceFE[]
  for block_name in block_names
    elem_name = mesh.element_types[block_name]
    elem_type = elem_type_map[uppercase(elem_name)]
    if p_degree === nothing
      p_degree = _default_p[uppercase(elem_name)]
    end

    if q_degree === nothing
      q_degree = _default_q[uppercase(elem_name)]
    end
    ref_fe = ReferenceFE(elem_type{interp_type, p_degree}(), q_type(q_degree))
    push!(ref_fes, ref_fe)
  end
  ref_fes = NamedTuple{tuple(Symbol.(values(block_names))...)}(tuple(ref_fes...))
  return ref_fes
end

function _setup_ref_fes(
  mesh, interp_type, p_degree, q_type, q_degree, ::Val{is_juliac_safe}
) where is_juliac_safe
  if is_juliac_safe
    return _juliac_safe_ref_fes
  else
    return _setup_ref_fes(mesh, interp_type, p_degree, q_type, q_degree)
  end
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
  IsJuliaCSafe,
  FieldType,
  IT            <: Integer,
  IV            <: AbstractVector{IT},
  RT            <: Number,
  RV            <: AbstractVector{RT},
  ND,
  BTRE,
  # Coords,
  RefFEs
} <: AbstractFunctionSpace
  block_names::Vector{String}
  block_to_ref_fe_id::BTRE
  coords::H1Field{RT, RV, ND}
  elem_conns::Connectivity{IT, IV}
  elem_id_maps::Vector{Vector{IT}} # TODO create new type for ID map similar to connectivity
  elem_to_facets::Union{Nothing, Connectivity{IT, IV}}
  facet_orientation::Union{Nothing, Connectivity{IT, IV}}
  # need to remove this mabye?
  node_id_map::IV
  ref_fes::RefFEs

  function FunctionSpace{is_juliac_safe, FT}(
    block_names, block_to_ref_fe_id, coords, elem_conns, 
    elem_id_maps, elem_to_facets, facet_orientation, node_id_map, ref_fes
  ) where {is_juliac_safe, FT}
    new{
      is_juliac_safe, FT, eltype(elem_conns.data), typeof(elem_conns.data), 
      eltype(coords), typeof(coords.data), size(coords, 1),
      typeof(block_to_ref_fe_id), typeof(ref_fes)
    }(
      block_names, block_to_ref_fe_id, coords, elem_conns, elem_id_maps,
      elem_to_facets, facet_orientation, node_id_map, ref_fes
    )
  end
end

function FunctionSpace(
  mesh::AbstractMesh, field_type::Type{FT}, interp_type,
  ::Type{QT} = GaussLobattoLegendre;
  is_juliac_safe::Bool = false,
  p_degree::Union{Int, Nothing} = nothing,
  q_degree::Union{Int, Nothing} = nothing,
) where {FT, QT <: ReferenceFiniteElements.AbstractQuadratureType}
  return FunctionSpace{is_juliac_safe}(mesh, field_type, interp_type, QT; p_degree = p_degree, q_degree = q_degree)
end

function FunctionSpace{is_juliac_safe}(
  mesh::AbstractMesh, ::Type{H1Field}, interp_type,
  q_type::Type{QT} = GaussLobattoLegendre;
  p_degree::Union{Int, Nothing} = nothing,
  q_degree::Union{Int, Nothing} = nothing,
) where {is_juliac_safe, QT <: ReferenceFiniteElements.AbstractQuadratureType}
  ref_fes = _setup_ref_fes(mesh, interp_type, p_degree, q_type, q_degree, Val{is_juliac_safe}())
  coords = mesh.nodal_coords
  conns = Connectivity([val for val in values(mesh.element_conns)])
  elem_id_maps = [val for val in values(mesh.element_id_maps)]
  block_to_ref_fe_id = _setup_block_to_ref_fe_id(mesh, Val{is_juliac_safe}())

  return FunctionSpace{is_juliac_safe, H1Field}(
    mesh.element_block_names, block_to_ref_fe_id, coords, 
    conns, elem_id_maps, nothing, nothing, mesh.node_id_map, ref_fes
  )
end

function FunctionSpace{is_juliac_safe}(
  mesh::AbstractMesh, ::Type{HdivField}, interp_type,
  q_type::Type{QT} = GaussLobattoLegendre;
  p_degree::Union{Int, Nothing} = nothing,
  q_degree::Union{Int, Nothing} = nothing,
) where {is_juliac_safe, QT <: ReferenceFiniteElements.AbstractQuadratureType}
  ref_fes = _setup_ref_fes(mesh, interp_type, p_degree, q_type, q_degree, Val{is_juliac_safe}())
  coords = mesh.nodal_coords
  conns = Connectivity([val for val in values(mesh.element_conns)])
  elem_id_maps = [val for val in values(mesh.element_id_maps)]
  block_to_ref_fe_id = _setup_block_to_ref_fe_id(mesh, Val{is_juliac_safe}())

  # TODO only will work for likely linear elements right now
  topology = UnstructuredTopology(mesh)
  elem_to_facets = Connectivity([val for val in values(topology.elem_to_facets)])
  facet_orientation = Connectivity([val for val in values(topology.facet_orientation)])

  return FunctionSpace{is_juliac_safe, HdivField}(
    mesh.element_block_names, block_to_ref_fe_id, coords, 
    conns, elem_id_maps, elem_to_facets, facet_orientation, mesh.node_id_map, ref_fes
  )
end

# TODO gonna have to fix this based on updates
function FunctionSpace{is_juliac_safe}(
  mesh::AbstractMesh, ::Type{L2Field}, interp_type,
  q_type::Type{QT} = GaussLobattoLegendre;
  p_degree = nothing,
  q_degree = nothing
) where {is_juliac_safe, QT <: ReferenceFiniteElements.AbstractQuadratureType}
  if interp_type == Lagrange && p_degree !== nothing && q_degree === nothing
    q_degree = p_degree + 1
  end
  ref_fes = _setup_ref_fes(mesh, interp_type, p_degree, q_type, q_degree, Val{is_juliac_safe}())
  conns = Connectivity([val for val in values(mesh.element_conns)])
  # coords = L2Field(map(x -> mesh.nodal_coords[:, x], [values(mesh.element_conns)...]))
  coords = mesh.nodal_coords
  # new_conns = Array{Int, 2}[]
  # offset = 1
  # for name in keys(mesh.element_conns)
  #   conn = mesh.element_conns[name]
  #   push!(new_conns, reshape(offset:offset + length(conn) - 1, size(conn)...))
  #   offset += size(conn, 1) * size(conn, 2)
  # end
  # conns = Connectivity(new_conns)
  conns = Connectivity([val for val in values(mesh.element_conns)])
  elem_id_maps = [val for val in values(mesh.element_id_maps)]
  block_to_ref_fe_id = _setup_block_to_ref_fe_id(mesh)

  return FunctionSpace{is_juliac_safe, L2Field}(
    mesh.element_block_names, block_to_ref_fe_id, coords,
    conns, elem_id_maps, nothing, nothing, mesh.node_id_map, ref_fes
  )
end

function Adapt.adapt_structure(to, fspace::FunctionSpace)
  return FunctionSpace{_is_juliac_safe(fspace), _field_type(fspace)}(
    fspace.block_names, fspace.block_to_ref_fe_id,
    adapt(to, fspace.coords),
    adapt(to, fspace.elem_conns), 
    # fspace.elem_id_maps,
    map(x -> adapt(to, x), fspace.elem_id_maps),
    adapt(to, fspace.elem_to_facets),
    adapt(to, fspace.facet_orientation),
    adapt(to, fspace.node_id_map),
    map(x -> adapt(to, x), fspace.ref_fes)
  )
end

function Base.show(io::IO, fspace::FunctionSpace)
  println(io, "FunctionSpace:")
  println(io, "  Type: $(typeof(fspace.coords).name.name)")
  for (block_name, block_id) in zip(fspace.block_names, fspace.block_to_ref_fe_id)
    println(io, "  $block_name:")
    # println(io, "    Element type = $(fspace.ref_fes[block_id].element)")
  end
end

function _field_type(::FunctionSpace{B, FT, I, V, BTRE, C, R}) where {B, FT, I, V, BTRE, C, R}
  return FT
end

function _is_juliac_safe(::FunctionSpace{B, FT, I, V, BTRE, C, R}) where {B, FT, I, V, BTRE, C, R}
  return B
end

function block_entity_size(fspace::FunctionSpace, b::Int)
  return (num_entities_per_element(fspace, b), num_elements(fspace, b))
end

function block_reference_element(fspace::FunctionSpace{false, FT, I, V, BTRE, C, R}, block_id::Int) where {FT, I, V, BTRE, C, R}
  return fspace.ref_fes[block_id]
end

@generated function block_reference_element(
  fspace::FunctionSpace{true, FT, IT, IV, BTRE, C, R},
  block_id::Int
) where {FT, IT, IV, BTRE, C, R}

  n_refs = length(BTRE.parameters)

  cases = map(1:n_refs) do j
    quote
      if block_id == $j
        return fspace.ref_fes[$j]
      end
    end
  end

  quote
    id = fspace.block_to_ref_fe_id[block_id]  # runtime value (OK)

    if id == -1
      error("Inactive block ", block_id)
    end

    $(cases...)  # ONLY depends on compile-time j

    error("Invalid ref_fe id ", id)
  end
end

# this does not work behind juliac
function block_quadrature_size(fspace::FunctionSpace{false}, b::Int)
  ref_fe = block_reference_element(fspace, b)
  nq = num_cell_quadrature_points(ref_fe)::Int
  ne = num_elements(fspace, b)
  return (nq, ne)
end

function block_quadrature_size(fspace::FunctionSpace{true}, b::Int)
  # this map allow us to actually have type stability for some reason
  nqs = map(x -> num_cell_quadrature_points(x), fspace.ref_fes)
  nq = nqs[fspace.block_to_ref_fe_id[b]]
  ne = num_elements(fspace, b)
  return (nq, ne)
end

function block_quadrature_sizes(fspace::FunctionSpace)
  sizes = Vector{Tuple{Int, Int}}(undef, num_blocks(fspace))
  for b in 1:num_blocks(fspace)
    sizes[b] = block_quadrature_size(fspace, b)
  end
  return sizes
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

function num_entities(fspace::FunctionSpace)
  if _field_type(fspace) == H1Field
    return size(fspace.coords, 2)
  elseif _field_type(fspace) == L2Field
    return mapreduce(
      x -> block_quadrature_size(fspace, x)[1] * block_quadrature_size(fspace, x)[2],
      prod, 1:num_blocks(fspace)
    )
  end
end

function num_entities_per_element(fspace::FunctionSpace, b::Int)
  if _field_type(fspace) == L2Field
    return block_quadrature_size(fspace, b)[1]
  else
    return num_entities_per_element(fspace.elem_conns, b)
  end
end

# function num_q_points(fspace::FunctionSpace, b::Int)
#   ref_fe = values(fspace.ref_fes)[b]
#   return num_quadrature_points(ref_fe)
# end

function unsafe_connectivity(fspace::FunctionSpace, e::Int, b::Int)
  return unsafe_connectivity(fspace.elem_conns, e, b)
end
