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
  Coords,
  ElemConns,
  RefFEs
} <: AbstractFunctionSpace
  coords::Coords
  elem_conns::ElemConns
  ref_fes::RefFEs
end

function _setup_ref_fes(mesh::AbstractMesh, interp_type, q_degree)
  block_names = mesh.element_block_names
  ref_fes = ReferenceFE[]
  for elem_name in mesh.element_types
    elem_type = elem_type_map[uppercase(elem_name)]
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
    mesh.element_conns,
    ref_fes
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
    nothing, # TODO what makes sense here?
    ref_fes
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

  coords_vals = L2QuadratureField.(coords_vals)
  coords = NamedTuple{tuple(coords_syms...)}(tuple(coords_vals...))

  return FunctionSpace(
    coords,
    nothing, # TODO what makes sense here?
    ref_fes
  )
end

function FunctionSpace(mesh::AbstractMesh, space_type, interp_type; q_degree=2)
  return FunctionSpace(mesh, space_type, interp_type, q_degree)
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

function map_shape_function_gradients(X, ∇N_ξ)
  J     = (X * ∇N_ξ)'
  J_inv = inv(J)
  ∇N_X  = (J_inv * ∇N_ξ')'
  return ∇N_X
end
