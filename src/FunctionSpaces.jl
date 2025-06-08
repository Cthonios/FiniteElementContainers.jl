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
  # Coords <: AbstractField, 
  Coords,
  ElemConns, ElemIdMaps, 
  # FSpaceType,
  RefFEs,
  SSetElems, SSetNodes, SSetSides, SSetSideNodes,
} <: AbstractFunctionSpace
  coords::Coords
  elem_conns::ElemConns
  elem_id_maps::ElemIdMaps
  # fspace_type::FSpaceType
  ref_fes::RefFEs
  sideset_elems::SSetElems
  sideset_nodes::SSetNodes
  sideset_sides::SSetSides
  sideset_side_nodes::SSetSideNodes
end

function _setup_ref_fes(mesh::AbstractMesh, interp_type, q_degree)
  block_names = mesh.element_block_names
  ref_fes = ReferenceFE[]
  for elem_name in mesh.element_types
    elem_type = elem_type_map[elem_name]
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

function FunctionSpace(mesh::AbstractMesh, ::Type{H1Field}, interp_type, q_degree)
  ref_fes = _setup_ref_fes(mesh, interp_type, q_degree)

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
    mesh.sideset_elems, 
    mesh.sideset_nodes, 
    mesh.sideset_sides, mesh.sideset_side_nodes
  )
end

# TODO this isn't correct currently. Need to have coords be the element centroids,
# not the collection of element level coordinates
function FunctionSpace(mesh::AbstractMesh, ::Type{L2ElementField}, interp_type, _)
  ref_fes = _setup_ref_fes(mesh, interp_type, 1) # 1 for element centroid. TODO will this work on all elements?

  coords_syms = Symbol[]
  coords_vals = Array{Float64, 3}[]

  # TODO fix this loop
  for (n, (block_name, conns)) in enumerate(pairs(mesh.element_conns))
    ref_fe = ref_fes[n]
    X_elems = mesh.nodal_coords[:, conns]
    coords_temp = _setup_quad_coords(mesh, X_elems, conns, ref_fe)
    # recyling L2QuadratureField method below, just need the field/element slices
    coords_temp = coords_temp[:, 1, :]
    push!(coords_syms, block_name)
    push!(coords_vals, X_elems)
  end

  coords = NamedTuple{tuple(coords_syms...)}(tuple(coords_vals...))

  return FunctionSpace(
    coords,
    nothing, nothing, # TODO what makes sense here?
    ref_fes,
    mesh.sideset_elems, 
    mesh.sideset_nodes, 
    mesh.sideset_sides, mesh.sideset_side_nodes
  )
end

# need to optimize this constructor
function FunctionSpace(mesh::AbstractMesh, ::Type{L2QuadratureField}, interp_type, q_degree)
  ref_fes = _setup_ref_fes(mesh, interp_type, q_degree)

  coords_syms = Symbol[]
  coords_vals = Array{Float64, 3}[]

  for (n, (block_name, conns)) in enumerate(pairs(mesh.element_conns))
    ref_fe = ref_fes[n]
    X_elems = mesh.nodal_coords[:, conns]
    coords_temp = _setup_quad_coords(mesh, X_elems, conns, ref_fe)
    push!(coords_syms, block_name)
    push!(coords_vals, coords_temp)
  end

  if num_fields(mesh.nodal_coords) == 1
    temp_syms = (:coords_x,)
  elseif num_fields(mesh.nodal_coords) == 2
    temp_syms = (:coords_x, :coords_y)
  else
    temp_syms = (:coords_x, :coords_y, :coords_z)
  end

  coords_vals = L2QuadratureField.(coords_vals, (temp_syms,))
  coords = NamedTuple{tuple(coords_syms...)}(tuple(coords_vals...))

  return FunctionSpace(
    coords,
    nothing, nothing, # TODO what makes sense here?
    ref_fes,
    mesh.sideset_elems, 
    mesh.sideset_nodes, 
    mesh.sideset_sides, mesh.sideset_side_nodes
  )
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
