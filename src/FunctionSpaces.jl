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

function _get_p_degree(el_type::String, p_degree::Union{Int, Nothing} = nothing)
  if p_degree === nothing
    return _default_p[el_type]
  else
    return p_degree
  end
end

function _get_q_degree(el_type::String, q_degree::Union{Int, Nothing} = nothing)
  if q_degree === nothing
    return _default_q[el_type]
  else
    return q_degree
  end
end

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

const MAX_BLOCKS = 16

function _setup_block_to_ref_fe_id(mesh::AbstractMesh, is_juliac_safe::Bool)
  if is_juliac_safe
    block_ids = Vector{Int}(undef, 0)
    for block_name in mesh.element_block_names
      el_type = mesh.element_types[block_name]
      push!(block_ids, _el_name_to_juliac_safe_id[el_type])
    end
    return block_ids
  else
    return 1:length(mesh.element_types) |> collect
  end
end

function _setup_juliac_safe_block_to_ref_fe_id(mesh::AbstractMesh)
  names = mesh.element_block_names
  el_types = map(x -> _el_name_to_juliac_safe_id[mesh.element_types[x]], names)
  N = length(names)
  return ntuple(i -> i <= N ? el_types[i] : -1, Val(MAX_BLOCKS))  # replace `i` with your actual block value
end

function _setup_block_to_ref_fe_id(mesh::AbstractMesh)
  return 1:length(mesh.element_types) |> collect
end

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

"""
default code path that sets up ref fes as a namedtuple
"""
function _setup_ref_fes(
  mesh::AbstractMesh, 
  interp_type, p_degree,
  q_type::Type{<:AbstractQuadratureType}, q_degree
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
  IsJuliaCSafe,
  IT            <: Integer,
  IV            <: AbstractVector{IT},
  BTRE,
  Coords,
  RefFEs
} <: AbstractFunctionSpace
  block_names::Vector{String}
  block_to_ref_fe_id::BTRE
  coords::Coords
  elem_conns::Connectivity{IT, IV}
  # need to remove this mabye?
  elem_id_maps::Vector{Vector{IT}} # TODO create new type for ID map similar to connectivity
  ref_fes::RefFEs

  function FunctionSpace{is_juliac_safe}(
    block_names, block_to_ref_fe_id, coords, conns, elem_id_maps, ref_fes
  ) where is_juliac_safe
    new{
      is_juliac_safe, eltype(conns.data), typeof(conns.data), 
      typeof(block_to_ref_fe_id), typeof(coords), typeof(ref_fes)
    }(block_names, block_to_ref_fe_id, coords, conns, elem_id_maps, ref_fes)
  end
end

function FunctionSpace(
  mesh::AbstractMesh, field_type::Type{IT}, interp_type,
  ::Type{QT} = GaussLobattoLegendre;
  is_juliac_safe::Bool = false,
  p_degree::Union{Int, Nothing} = nothing,
  q_degree::Union{Int, Nothing} = nothing,
) where {IT, QT <: AbstractQuadratureType}
  return FunctionSpace{is_juliac_safe}(mesh, field_type, interp_type, QT; p_degree = p_degree, q_degree = q_degree)
end

function FunctionSpace{is_juliac_safe}(
  mesh::AbstractMesh, ::Type{H1Field}, interp_type,
  q_type::Type{QT} = GaussLobattoLegendre;
  p_degree::Union{Int, Nothing} = nothing,
  q_degree::Union{Int, Nothing} = nothing,
) where {is_juliac_safe, QT <: AbstractQuadratureType}
  # TODO move to some common function so we can use it across
  # all constructors
  if p_degree !== nothing
    # some error checking on p_degree input
    if p_degree < 0
      @assert false "Bad polynomial degree $(p_degree)"
    elseif p_degree == 0
      @assert false "TODO 0 order elements"
    else
      # TODO all of this needs TLC
      ref_fes = _setup_ref_fes(mesh, interp_type, p_degree, q_type, q_degree)
      coords, conns = create_higher_order_mesh(mesh, H1Field, interp_type, p_degree)
      conns = Connectivity([val for val in values(conns)])
    end
  else
    if is_juliac_safe
      # ref_fes = _setup_juliac_safe_ref_fes(interp_type, q_type)
      ref_fes = _juliac_safe_ref_fes
    else
      ref_fes = _setup_ref_fes(mesh, interp_type, nothing, q_type, q_degree)
    end
    coords = mesh.nodal_coords
    conns = Connectivity([val for val in values(mesh.element_conns)])
  end
  elem_id_maps = [val for val in values(mesh.element_id_maps)]
  if is_juliac_safe
    block_to_ref_fe_id = _setup_juliac_safe_block_to_ref_fe_id(mesh)
  else
    # block_to_ref_fe_id = _setup_block_to_ref_fe_id(mesh, is_juliac_afe)
    block_to_ref_fe_id = _setup_block_to_ref_fe_id(mesh)
  end

  return FunctionSpace{is_juliac_safe}(mesh.element_block_names, block_to_ref_fe_id, coords, conns, elem_id_maps, ref_fes)
end

# TODO gonna have to fix this based on updates
function FunctionSpace{is_juliac_safe}(
  mesh::AbstractMesh, ::Type{L2Field}, interp_type,
  q_type::Type{QT} = GaussLobattoLegendre;
  p_degree = nothing,
  q_degree = nothing
) where {is_juliac_safe, QT <: AbstractQuadratureType}
  if is_juliac_safe
    ref_fes = _setup_juliac_safe_ref_fes(interp_type, q_type)
  else
    ref_fes = _setup_ref_fes(mesh, interp_type, p_degree, q_type, q_degree)
  end

  conns = Connectivity([val for val in values(mesh.element_conns)])
  coords = L2Field(map(x -> mesh.nodal_coords[:, x], [values(mesh.element_conns)...]))

  new_conns = Array{Int, 2}[]
  offset = 1
  for name in keys(mesh.element_conns)
    conn = mesh.element_conns[name]
    push!(new_conns, reshape(offset:offset + length(conn) - 1, size(conn)...))
    offset += size(conn, 1) * size(conn, 2)
  end
  conns = Connectivity(new_conns)
  elem_id_maps = [val for val in values(mesh.element_id_maps)]
  block_to_ref_fe_id = _setup_block_to_ref_fe_id(mesh, is_juliac_safe)

  return FunctionSpace{is_juliac_safe}(mesh.element_block_names, block_to_ref_fe_id, coords, conns, elem_id_maps, ref_fes)
end

function Adapt.adapt_structure(to, fspace::FunctionSpace)
  return FunctionSpace{_is_juliac_safe(fspace)}(
    fspace.block_names, fspace.block_to_ref_fe_id,
    adapt(to, fspace.coords),
    adapt(to, fspace.elem_conns), 
    # fspace.elem_id_maps,
    map(x -> adapt(to, x), fspace.elem_id_maps),
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

function _is_juliac_safe(::FunctionSpace{B, I, V, BTRE, C, R}) where {B, I, V, BTRE, C, R}
  return B
end

function block_entity_size(fspace::FunctionSpace, b::Int)
  return (num_entities_per_element(fspace, b), num_elements(fspace, b))
end

function block_reference_element(fspace::FunctionSpace{false, I, V, BTRE, C, R}, block_id::Int) where {I, V, BTRE, C, R}
  return fspace.ref_fes[block_id]
end

@generated function block_reference_element(
  fspace::FunctionSpace{true, IT, IV, BTRE, C, R},
  block_id::Int
) where {IT, IV, BTRE, C, R}

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

# function block_reference_element(
#   fspace::FunctionSpace{true, IT, IV, BTRE, C, R},
#   block_id::Int
# ) where {IT, IV, BTRE, C, R}
#   index = fspace.block_to_ref_fe_id[block_id]
#   return _block_reference_element(fspace, Val{index}())
# end

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

function num_entities_per_element(fspace::FunctionSpace, b::Int)
  return num_entities_per_element(fspace.elem_conns, b)
end

# function num_q_points(fspace::FunctionSpace, b::Int)
#   ref_fe = values(fspace.ref_fes)[b]
#   return num_quadrature_points(ref_fe)
# end

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

