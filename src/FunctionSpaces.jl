elem_type_map = Dict{String, Type{<:ReferenceFiniteElements.ReferenceFEType}}(
  "HEX"     => Hex8,
  "HEX8"    => Hex8,
  "QUAD"    => Quad4,
  "QUAD4"   => Quad4,
  "QUAD9"   => Quad9,
  "TRI"     => Tri3,
  "TRI3"    => Tri3,
  "TRI6"    => Tri6,
  "TET"     => Tet4,
  "TETRA4"  => Tet4,
  "TETRA10" => Tet10
)

# elem_type = @NamedTuple begin
#   TETRA4::Type{Tet4}
#   TETRA10::Type{Tet10}
# end
# elem_type_map = elem_type((
#   Tet4,
#   Tet10
# ))

abstract type AbstractFunctionSpace{Conn, RefFE} <: FEMContainer end
connectivity(fspace::AbstractFunctionSpace)          = fspace.conn
reference_element(fspace::AbstractFunctionSpace)     = fspace.ref_fe
num_dimensions(fspace::AbstractFunctionSpace)        = ReferenceFiniteElements.num_dimensions(fspace.ref_fe.ref_fe_type)
num_elements(fspace::AbstractFunctionSpace)          = num_elements(fspace.conn)
num_nodes_per_element(fspace::AbstractFunctionSpace) = ReferenceFiniteElements.num_nodes_per_element(fspace.ref_fe)
num_q_points(fspace::AbstractFunctionSpace)          = ReferenceFiniteElements.num_q_points(fspace.ref_fe)

quadrature_point(fspace::AbstractFunctionSpace, q::Int) =
ReferenceFiniteElements.quadrature_point(fspace.ref_fe, q)

quadrature_weight(fspace::AbstractFunctionSpace, q::Int) = 
ReferenceFiniteElements.quadrature_weight(fspace.ref_fe, q)

shape_function_values(fspace::AbstractFunctionSpace, q::Int) = 
ReferenceFiniteElements.shape_function_values(fspace.ref_fe, q) 

shape_function_gradients(fspace::AbstractFunctionSpace, q::Int) = 
ReferenceFiniteElements.shape_function_gradients(fspace.ref_fe, q)

element_level_fields(fspace::AbstractFunctionSpace, u::NodalField)         = element_level_fields(fspace.conn, u)
element_level_fields(fspace::AbstractFunctionSpace, e::Int, u::NodalField) = element_level_fields(fspace.conn, e, u)

element_level_fields_reinterpret(fspace::AbstractFunctionSpace, u::NodalField) = 
element_level_fields_reinterpret(fspace.conn, u)

function volume(X, ∇N_ξ)
  J = X * ∇N_ξ
  return det(J)
end

function map_shape_function_gradients(X, ∇N_ξ)
  J     = X * ∇N_ξ
  J_inv = inv(J)
  ∇N_X  = (J_inv * ∇N_ξ')'
  return ∇N_X
end

# function setup_reference_element(type::Symbol, q_degree)
#   el_type = elem_type_map[type]
#   return setup_reference_element(el_type, q_degree)
# end

function setup_reference_element(type::Type{<:ReferenceFiniteElements.ReferenceFEType}, q_degree)
  ReferenceFiniteElements.ReferenceFE(type(Val(q_degree)))
end



#######################################################

# still need more work on this one. Big questions are what to do with adjoints
# and discrete gradient operators that will get bigger to store

struct NonAllocatedFunctionSpace{
  Conn  <: Connectivity,
  RefFE <: ReferenceFE
} <: AbstractFunctionSpace{Conn, RefFE}

  conn::Conn
  ref_fe::RefFE
end

function NonAllocatedFunctionSpace(mesh::Mesh, block_id::Int, q_degree::Int)
  conn      = mesh.conns[block_id]
  elem_type = mesh.elem_types[block_id]
  ref_fe    = ReferenceFE(elem_type_map[elem_type](Val(q_degree)))
  @assert num_nodes_per_element(conn) == ReferenceFiniteElements.num_nodes_per_element(ref_fe)
  return NonAllocatedFunctionSpace(conn, ref_fe)
end

#######################################################

struct Interpolants{A1, A2, A3, A4}
  X_q::A1
  N::A2
  ∇N_X::A3
  JxW::A4
end

# TODO try to move this to an abstract method
function Base.getindex(fspace::NonAllocatedFunctionSpace, q::Int, e::Int, X::NodalField)
  X_el = element_level_fields(fspace, e, X) 
  N    = shape_function_values(fspace, q)
  ∇N_ξ = shape_function_gradients(fspace, q)
  ∇N_X = map_shape_function_gradients(X_el, ∇N_ξ)
  JxW  = volume(X_el, ∇N_ξ) * quadrature_weight(fspace, q)
  X_q  = X_el * N
  # return X_q, N, ∇N_X, JxW
  return Interpolants(X_q, N, ∇N_X, JxW)
end

struct AllocatedFunctionSpace{
  Conn  <: Connectivity,
  RefFE <: ReferenceFE,
  S     <: StructArray
} <: AbstractFunctionSpace{Conn, RefFE}

  conn::Conn
  ref_fe::RefFE
  interps::S
end

function AllocatedFunctionSpace!(X_qs, Ns, ∇N_Xs, JxWs, Xs, conn, ref_fe)
  for e in axes(X_qs, 2)
    X_e = element_level_fields(conn, e, Xs)
    for q in axes(X_qs, 1)
      N    = ReferenceFiniteElements.shape_function_values(ref_fe, q)
      ∇N_ξ = ReferenceFiniteElements.shape_function_gradients(ref_fe, q)

      X_q  = X_e * N
      ∇N_X = map_shape_function_gradients(X_e, ∇N_ξ)
      JxW  = volume(X_e, ∇N_ξ) * ReferenceFiniteElements.quadrature_weight(ref_fe, q)

      X_qs[q, e]  = X_q
      Ns[q, e]    = N
      ∇N_Xs[q, e] = ∇N_X
      JxWs[q, e]  = JxW
    end
  end
end

# TODO make constructor parameteric on type
function AllocatedFunctionSpace(mesh::Mesh, block_id::Int, q_degree)
  # TODO maybe move below to a common function since it's exactly the same
  # as the constructor for NonAllocatedFunctionSpace
  conn      = mesh.conns[block_id]
  elem_type = elem_type_map[mesh.elem_types[block_id]] #|> Symbol
  ref_fe = setup_reference_element(elem_type, q_degree)

  # error checking
  @assert num_dimensions(mesh)        == ReferenceFiniteElements.num_dimensions(ref_fe.ref_fe_type)
  @assert num_nodes_per_element(conn) == ReferenceFiniteElements.num_nodes_per_element(ref_fe)

  # TODO need to type perly below for Quantities eventually
  # initialize arrays
  T     = eltype(mesh.coords)
  N, D  = num_nodes_per_element(conn), num_dimensions(mesh)
  E, Q  = num_elements(conn), ReferenceFiniteElements.num_q_points(ref_fe)

  X_qs  = Matrix{SVector{D, eltype(mesh.coords)}}(undef, Q, E)
  Ns    = Matrix{eltype(ref_fe.interpolants.N)}(undef, Q, E)
  ∇N_Xs = Matrix{eltype(ref_fe.interpolants.∇N_ξ)}(undef, Q, E)
  JxWs  = Matrix{eltype(mesh.coords)}(undef, Q, E)

  AllocatedFunctionSpace!(X_qs, Ns, ∇N_Xs, JxWs, mesh.coords, conn, ref_fe)

  interps = StructArray{Interpolants{eltype(X_qs), eltype(Ns), eltype(∇N_Xs), eltype(JxWs)}}((X_qs, Ns, ∇N_Xs, JxWs))

  # TODO convert to element field and give it names and the total number of fields equal to number
  # of total fields in the structarray
  # interps = ElementField{}
  return AllocatedFunctionSpace(conn, ref_fe, interps)
end

"""
In place method to update the quadrature values given
a function space and the element level values
"""
function field_values!(u_qs, fspace::NonAllocatedFunctionSpace, u_els)
  for e in axes(u_qs, 2)
    for q in axes(u_qs, 1)
      u_qs[q, e] = u_els[e] * shape_function_values(fspace, q)
    end
  end
end

"""
In place method to update quadrature values given
a function space, connectivity, and the 
the nodal values
"""
function field_values!(u_qs, fspace::NonAllocatedFunctionSpace, ::NodalField, u::NodalField)
  u_els = element_level_fields_reinterpret(fspace.conn, u)
  for e in axes(u_qs, 2)
    for q in axes(u_qs, 1)
      u_qs[q, e] = u_els[e] * shape_function_values(fspace, q)
    end
  end
end

"""
Out of place method to update the quadrature values
"""
function field_values(fspace::NonAllocatedFunctionSpace, X::NodalField, u::NodalField)
  NF    = num_fields(u)
  NQ    = num_q_points(fspace)
  NE    = num_elements(fspace)
  T     = SVector{NF, eltype(u)}
  u_els = element_level_fields_reinterpret(fspace, u)
  u_qs  = QuadratureField{NF, NQ, NE}(StructArray, T, Symbol("quadrature_values_", field_names(u)), undef)
  field_values!(u_qs, fspace, u_els)
  return u_qs
end

# Too many allocations
# function field_values_v2(fspace::NonAllocatedFunctionSpace, X::NodalField, u::NodalField)
#   X_els = element_level_fields_reinterpret(fspace.conn, X)
#   return ElementField{length(X_els[1]), length(X_els)}(X_els, Symbol("element_level"))
# end

function field_values(fspace::NonAllocatedFunctionSpace, q::Int, e::Int, ::NodalField, u::NodalField)
  u_el = element_level_fields(fspace, e, u)
  N_q  = shape_function_values(fspace, q)
  u_q  = u_el * N_q
  return u_q
end


#########################################

function field_gradients!(∇u_qs, fspace::NonAllocatedFunctionSpace, X_els, u_els)
  for e in axes(∇u_qs, 2)
    for q in axes(∇u_qs, 1)
      ∇N_ξ = shape_function_gradients(fspace, q)
      ∇N_X = map_shape_function_gradients(X_els[e], ∇N_ξ)
      ∇u_qs[q, e] = u_els[e] * ∇N_X
    end
  end
end

function field_gradients!(∇u_qs, fspace::NonAllocatedFunctionSpace, X::NodalField, u::NodalField)
  X_els = element_level_fields_reinterpret(fspace, X)
  u_els = element_level_fields_reinterpret(fspace, u)
  for e in axes(∇u_qs, 2)
    for q in axes(∇u_qs, 1)
      ∇N_ξ = shape_function_gradients(fspace, q)
      ∇N_X = map_shape_function_gradients(X_els[e], ∇N_ξ)
      ∇u_qs[q, e] = u_els[e] * ∇N_X
    end
  end
end

function field_gradients(fspace::NonAllocatedFunctionSpace, X::NodalField, u::NodalField)
  NF    = num_fields(u)
  D     = num_dimensions(fspace)
  NQ    = num_q_points(fspace)
  NE    = num_elements(fspace)
  T     = SMatrix{NF, D, eltype(u), NF * D}
  X_els = element_level_fields_reinterpret(fspace, X)
  u_els = element_level_fields_reinterpret(fspace, u)
  ∇u_qs = QuadratureField{NF * D, NQ, NE}(StructArray, T, Symbol("quadrature_gradients_", field_names(u)), undef)
  field_gradients!(∇u_qs, fspace, X_els, u_els)
  return ∇u_qs
end

function field_gradients(fspace::NonAllocatedFunctionSpace, q::Int, e::Int, X::NodalField, u::NodalField)
  X_el = element_level_fields(fspace, e, X)
  u_el = element_level_fields(fspace, e, u)
  ∇N_ξ = shape_function_gradients(fspace, q)
  ∇N_X = map_shape_function_gradients(X_el, ∇N_ξ)
  ∇u_q = u_el * ∇N_X
  return ∇u_q
end

# ###############################################################

function JxW!(JxWs, fspace::NonAllocatedFunctionSpace, X_els)
  for e in axes(JxWs, 2)
    for q in axes(JxWs, 1)
      ∇N_ξ = shape_function_gradients(fspace, q)
      detJ = volume(X_els[e], ∇N_ξ)
      w    = quadrature_weight(fspace, q)
      JxWs[q, e] = detJ * w
    end
  end
end

function JxW(fspace::NonAllocatedFunctionSpace, X::NodalField)
  NQ    = num_q_points(fspace)
  NE    = num_elements(fspace)
  T     = eltype(X)
  X_els = element_level_fields_reinterpret(fspace, X)
  JxWs  = QuadratureField{1, NQ, NE}(Matrix, T, Symbol("JxW"), undef)
  JxW!(JxWs, fspace, X_els)
  return JxWs
end

function JxW(fspace::NonAllocatedFunctionSpace, q::Int, e::Int, X::NodalField)
  X_el = element_level_fields(fspace, e, X)
  ∇N_ξ = shape_function_gradients(fspace, q)
  detJ = volume(X_el, ∇N_ξ)
  w    = quadrature_weight(fspace, q)
  return detJ * w
end

volume(fspace::NonAllocatedFunctionSpace, X::NodalField) = sum(JxW(fspace, X)) 

################################################################################



##############################################################################

abstract type AbstractMechanicsFormulation end

struct PlaneStrain <: AbstractMechanicsFormulation
end

struct ThreeDimensional <: AbstractMechanicsFormulation
end

# for large tuples, otherwise we get allocations
@generated function set(t::Tuple{Vararg{Any, N}}, x, i) where {N}
  Expr(:tuple, (:(ifelse($j == i, x, t[$j])) for j in 1:N)...)
end

function discrete_gradient(::Type{PlaneStrain}, ∇N_X)
  N   = size(∇N_X, 1)
  tup = ntuple(i -> 0.0, Val(4 * 2 * N))

  for n in 1:N
    k = 2 * (n - 1) 
    tup = setindex(tup, ∇N_X[n, 1], k + 1)
    tup = setindex(tup, 0.0,        k + 2)

    k = 2 * (n - 1) + 2 * N
    tup = setindex(tup, 0.0,        k + 1)
    tup = setindex(tup, ∇N_X[n, 1], k + 2)
    
    k = 2 * (n - 1)  + 2 * 2 * N
    tup = setindex(tup, ∇N_X[n, 2], k + 1)
    tup = setindex(tup, 0.0,        k + 2)

    k = 2 * (n - 1) + 3 * 2 * N
    tup = setindex(tup, 0.0,        k + 1)
    tup = setindex(tup, ∇N_X[n, 2], k + 2)
  end

  return SMatrix{2 * N, 4, eltype(∇N_X), 2 * N * 4}(tup)
end

function discrete_gradient(::Type{ThreeDimensional}, ∇N_X)
  N   = size(∇N_X, 1)
  tup = ntuple(i -> 0.0, Val(9 * 3 * N))

  for n in 1:N
    k = 3 * (n - 1) 
    tup = set(tup, ∇N_X[n, 1], k + 1)
    tup = set(tup, 0.0,        k + 2)
    tup = set(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 3 * N
    tup = set(tup, 0.0,        k + 1)
    tup = set(tup, ∇N_X[n, 1], k + 2)
    tup = set(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 2 * 3 * N
    tup = set(tup, 0.0,        k + 1)
    tup = set(tup, 0.0,        k + 2)
    tup = set(tup, ∇N_X[n, 1], k + 3)

    #
    k = 3 * (n - 1) + 3 * 3 * N
    tup = set(tup, ∇N_X[n, 2], k + 1)
    tup = set(tup, 0.0,        k + 2)
    tup = set(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 4 * 3 * N
    tup = set(tup, 0.0,        k + 1)
    tup = set(tup, ∇N_X[n, 2], k + 2)
    tup = set(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 5 * 3 * N
    tup = set(tup, 0.0,        k + 1)
    tup = set(tup, 0.0,        k + 2)
    tup = set(tup, ∇N_X[n, 2], k + 3)

    #
    k = 3 * (n - 1) + 6 * 3 * N
    tup = set(tup, ∇N_X[n, 3], k + 1)
    tup = set(tup, 0.0,        k + 2)
    tup = set(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 7 * 3 * N
    tup = set(tup, 0.0,        k + 1)
    tup = set(tup, ∇N_X[n, 3], k + 2)
    tup = set(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 8 * 3 * N
    tup = set(tup, 0.0,        k + 1)
    tup = set(tup, 0.0,        k + 2)
    tup = set(tup, ∇N_X[n, 3], k + 3)
  end

  return SMatrix{3 * N, 9, eltype(∇N_X), 3 * N * 9}(tup)
end

function discrete_gradient(fspace::NonAllocatedFunctionSpace, type::Type{PlaneStrain}, q, e, X)
  D = num_dimensions(fspace)

  @assert D == 2

  X_el = element_level_fields(fspace, e, X)
  ∇N_X = map_shape_function_gradients(X_el, shape_function_gradients(fspace, q))
  G    = discrete_gradient(type, ∇N_X)
  return G
end

function discrete_gradient(fspace::AllocatedFunctionSpace, type::Type{PlaneStrain}, q, e, X)
  ∇N_X = fspace.∇N_X[q, e]
  G    = discrete_gradient(type, ∇N_X)
  return G
end

function discrete_gradient(fspace, type::Type{ThreeDimensional}, q, e, X)
  N, D = num_nodes_per_element(fspace), num_dimensions(fspace)

  @assert D == 3

  X_el = element_level_fields(fspace, e, X)
  ∇N_X = map_shape_function_gradients(X_el, shape_function_gradients(fspace, q))
  G    = discrete_gradient(type, ∇N_X)
  return G
end

function discrete_symmetric_gradient(::Type{PlaneStrain}, ∇N_X)
  N   = size(∇N_X, 1)
  tup = ntuple(i -> 0.0, Val(3 * 2 * N))

  for n in 1:N
    k = 2 * (n - 1) 
    tup = setindex(tup, ∇N_X[n, 1], k + 1)
    tup = setindex(tup, 0.0,        k + 2)

    k = 2 * (n - 1) + 2 * N
    tup = setindex(tup, 0.0,        k + 1)
    tup = setindex(tup, ∇N_X[n, 2], k + 2)

    k = 2 * (n - 1) + 2 * 2 * N
    tup = setindex(tup, ∇N_X[n, 2], k + 1)
    tup = setindex(tup, ∇N_X[n, 1], k + 2)
  end

  return SMatrix{2 * N, 3, eltype(∇N_X), 2 * N * 3}(tup)
end

function discrete_symmetric_gradient(::Type{ThreeDimensional}, ∇N_X)
  N   = size(∇N_X, 1)
  tup = ntuple(i -> 0.0, Val(6 * 3 * N))

  for n in 1:N
    k = 3 * (n - 1)
    tup = set(tup, ∇N_X[n, 1], k + 1)
    tup = set(tup, 0.0,        k + 2)
    tup = set(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 3 * N
    tup = set(tup, 0.0,        k + 1)
    tup = set(tup, ∇N_X[n, 2], k + 2)
    tup = set(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 2 * 3 * N
    tup = set(tup, 0.0,        k + 1)
    tup = set(tup, 0.0,        k + 2)
    tup = set(tup, ∇N_X[n, 3], k + 3)

    k = 3 * (n - 1) + 3 * 3 * N
    tup = set(tup, ∇N_X[n, 2], k + 1)
    tup = set(tup, ∇N_X[n, 1], k + 2)
    tup = set(tup, 0.0,        k + 3)

    k = 3 * (n - 1) + 4 * 3 * N
    tup = set(tup, 0.0,        k + 1)
    tup = set(tup, ∇N_X[n, 3], k + 2)
    tup = set(tup, ∇N_X[n, 2], k + 3)

    k = 3 * (n - 1) + 5 * 3 * N
    tup = set(tup, ∇N_X[n, 3], k + 1)
    tup = set(tup, 0.0,        k + 2)
    tup = set(tup, ∇N_X[n, 1], k + 3)

  end
  return SMatrix{3 * N, 6, eltype(∇N_X), 3 * N * 6}(tup)
end

function discrete_symmetric_gradient(fspace, type::Type{<:AbstractMechanicsFormulation}, q, e, X)
  D = num_dimensions(fspace)

  # @assert D == 2
  if type <: PlaneStrain
    @assert D == 2
  elseif type <: ThreeDimensional
    @assert D == 3
  else
    @assert false
  end

  X_el = element_level_fields(fspace, e, X)
  ∇N_X = map_shape_function_gradients(X_el, shape_function_gradients(fspace, q))
  G    = discrete_symmetric_gradient(type, ∇N_X)
  return G
end