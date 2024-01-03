# elem_type_map = Dict{String, Type{<:ReferenceFiniteElements.ReferenceFEType}}(
#   "HEX"     => Hex8,
#   "HEX8"    => Hex8,
#   "QUAD"    => Quad4,
#   "QUAD4"   => Quad4,
#   "QUAD9"   => Quad9,
#   "TRI"     => Tri3,
#   "TRI3"    => Tri3,
#   "TRI6"    => Tri6,
#   "TET"     => Tet4,
#   "TETRA4"  => Tet4,
#   "TETRA10" => Tet10
# )

# #####################################################

# function volume(X, ∇N_ξ)
#   J = X * ∇N_ξ
#   return det(J)
# end

# function map_shape_function_gradients(X, ∇N_ξ)
#   J     = X * ∇N_ξ
#   J_inv = inv(J)
#   ∇N_X  = (J_inv * ∇N_ξ')'
#   return ∇N_X
# end

# function setup_reference_element(
#   type::Type{<:ReferenceFiniteElements.ReferenceFEType}, 
#   q_degree
# )
#   ReferenceFiniteElements.ReferenceFE(type(Val(q_degree)))
# end

# function setup_dof_connectivity!(dof_conns, ids, conns)
#   for e in 1:num_elements(dof_conns)
#     conn            = connectivity(conns, e)
#     # dof_conns[:, e] = ids[:, vec(conn)]
#     dof_conns[:, e] = ids[:, conn]
#   end
# end

include("function_spaces/Utils.jl")

#####################################################

abstract type FunctionSpace{NDof, RefFE, Conn} end
connectivity(fspace::FunctionSpace)             = fspace.conn
connectivity(fspace::FunctionSpace, e::Int)     = connectivity(fspace.conn, e)
dof_connectivity(fspace::FunctionSpace)         = fspace.dof_conn
dof_connectivity(fspace::FunctionSpace, e::Int) = connectivity(fspace.dof_conn, e)
reference_element(fspace::FunctionSpace)        = fspace.ref_fe
num_dimensions(fspace::FunctionSpace)           = ReferenceFiniteElements.num_dimensions(fspace.ref_fe.ref_fe_type)
num_elements(fspace::FunctionSpace)             = num_elements(fspace.conn)
num_nodes_per_element(fspace::FunctionSpace)    = ReferenceFiniteElements.num_nodes_per_element(fspace.ref_fe)
num_q_points(fspace::FunctionSpace)             = ReferenceFiniteElements.num_q_points(fspace.ref_fe)
num_dofs_per_node(::FunctionSpace{ND, RefFE, Conn}) where {ND, RefFE, Conn} = ND

#####################################################

quadrature_point(fspace::FunctionSpace, q::Int) =
ReferenceFiniteElements.quadrature_point(fspace.ref_fe, q)

quadrature_weight(fspace::FunctionSpace, q::Int) = 
ReferenceFiniteElements.quadrature_weight(fspace.ref_fe, q)

shape_function_values(fspace::FunctionSpace, q::Int) = 
ReferenceFiniteElements.shape_function_values(fspace.ref_fe, q) 

shape_function_gradients(fspace::FunctionSpace, q::Int) = 
ReferenceFiniteElements.shape_function_gradients(fspace.ref_fe, q)

#####################################################

function element_level_fields(fspace::FunctionSpace, u::NodalField)
  u_el = element_level_fields_reinterpret(fspace, u)
  NN, NE = num_nodes_per_element(fspace), num_elements(fspace)
  u_el_ret = ElementField{NN, NE, StructVector, eltype(u_el)}(undef)
  u_el_ret .= u_el
end

function element_level_fields(fspace::FunctionSpace, u, e::Int)
  NF, NN = num_fields(u), num_nodes_per_element(fspace)
  u_el = SMatrix{NF, NN, eltype(u), NF * NN}(@views vec(u[:, connectivity(fspace, e)]))
  return u_el
end

function element_level_fields_reinterpret(fspace::FunctionSpace, u::NodalField)
  NF, NN = num_fields(u), num_nodes_per_element(fspace)
  u_el = reinterpret(SMatrix{NF, NN, eltype(u), NF * NN}, @views vec(u[:, connectivity(fspace)]))
  return u_el
end

function element_level_fields_reinterpret(fspace::FunctionSpace, u::NodalField, e::Int)
  NF, NN = num_fields(u), num_nodes_per_element(fspace)
  u_el = reinterpret(SMatrix{NF, NN, eltype(u), NF * NN}, @views vec(u[:, connectivity(fspace, e)]))
  return u_el
end

function quadrature_level_field_values(fspace::FunctionSpace, ::NodalField, u::NodalField, q::Int, e::Int)
  u_el = element_level_fields(fspace, u, e)
  N_q  = shape_function_values(fspace, q)
  return u_el * N_q
end

function quadrature_level_field_gradients(fspace::FunctionSpace, X::NodalField, u::NodalField, q::Int, e::Int)
  X_el = element_level_fields(fspace, X, e)
  u_el = element_level_fields(fspace, u, e)
  ∇N_ξ = shape_function_gradients(fspace, q)
  ∇N_X = map_shape_function_gradients(X_el, ∇N_ξ)
  return u_el * ∇N_X
end

volume(fspace::FunctionSpace, X::NodalField, q::Int, e::Int) = fspace[X, q, e].JxW
function volume(fspace::FunctionSpace, X::NodalField, e::Int)
  v = 0.0 # TODO place for unitful to not work
  for q in 1:num_q_points(fspace)
    v = v + volume(fspace, X, q, e)
  end
  return v
end

function volume(fspace::FunctionSpace, X::NodalField)
  v = 0.0
  for e in 1:num_elements(fspace)
    for q in 1:num_q_points(fspace)
      v = v + volume(fspace, X, q, e)
    end
  end
  return v
end

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

# function discrete_gradient_v2(::Type{PlaneStrain}, ∇N_X)
#   N = size(∇N_X, 1)

#   SMatrix{2 * N, 4, eltype(∇N_X), 2 * N * 4}(
#     begin
#       if i == 1
#         if j % 2 != 0
#           ∇N_X[div(j, 2) + 1, 1]
#         else
#           0
#         end
#       elseif i == 2
#         if j % 2 != 0
#           0
#         else
#           ∇N_X[div(j, 2), 1]
#         end
#       else
#         -1
#       end
#     end for j = 1:2 * N, i = 1:4
#   )
# end

function discrete_gradient(::PlaneStrain, ∇N_X)
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

function discrete_gradient(fspace::FunctionSpace, type::Type{<:AbstractMechanicsFormulation}, X, q, e)
  D = num_dimensions(fspace)

  if type <: PlaneStrain
    @assert D == 2
  elseif type <: ThreeDimensional
    @assert D == 3
  else
    @assert false
  end

  X_el = element_level_fields(fspace, X, e)
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

function discrete_symmetric_gradient(fspace, type::Type{<:AbstractMechanicsFormulation}, X, q, e)
  D = num_dimensions(fspace)

  # @assert D == 2
  if type <: PlaneStrain
    @assert D == 2
  elseif type <: ThreeDimensional
    @assert D == 3
  else
    @assert false
  end

  X_el = element_level_fields(fspace, X, e)
  ∇N_X = map_shape_function_gradients(X_el, shape_function_gradients(fspace, q))
  G    = discrete_symmetric_gradient(type, ∇N_X)
  return G
end

#####################################################

function modify_field_gradients(::PlaneStrain, ∇u_q::SMatrix{2, 2, T, 4}, ::Type{<:Tensor}) where T <: Number
  return Tensor{2, 3, T, 9}((
    ∇u_q[1, 1], ∇u_q[2, 1], 0.0,
    ∇u_q[1, 2], ∇u_q[2, 2], 0.0,
    0.0,        0.0,        0.0
  ))
end

function modify_field_gradients(::PlaneStrain, ∇u_q::SMatrix{2, 2, T, 4}, ::Type{<:SArray}) where T <: Number
  return SMatrix{3, 3, T, 9}((
    ∇u_q[1, 1], ∇u_q[2, 1], 0.0,
    ∇u_q[1, 2], ∇u_q[2, 2], 0.0,
    0.0,        0.0,        0.0
  ))
end

modify_field_gradients(form::PlaneStrain, ∇u_q::SMatrix{2, 2, T, 4}; type = Tensor) where T <: Number =
modify_field_gradients(form, ∇u_q, type)

function modify_field_gradients(::PlaneStrain, ∇u_q::Tensor{2, 2, T, 4}, ::Type{<:Tensor}) where T <: Number
  return Tensor{2, 3, T, 9}((
    ∇u_q[1, 1], ∇u_q[2, 1], 0.0,
    ∇u_q[1, 2], ∇u_q[2, 2], 0.0,
    0.0,        0.0,        0.0
  ))
end

function extract_stress(::PlaneStrain, P::Tensor{2, 3, T, 9}) where T <: Number
  P_vec = tovoigt(SVector, P)
  # return SVector{4, T}((P_vec[1], P_vec[2], P_vec[6], P_vec[9]))
  # return SVector{4, T}((P_vec[1], P_vec[2], P_vec[9], P_vec[6]))
  return SVector{4, T}((P_vec[1], P_vec[9], P_vec[6], P_vec[2]))
end

function extract_stiffness(::PlaneStrain, A::Tensor{4, 3, T, 81}) where T <: Number
  A_mat = tovoigt(SMatrix, A)
  # return SMatrix{4, 4, T, 16}((
  #   A_mat[1, 1], A_mat[2, 1], A_mat[6, 1], A_mat[9, 1],
  #   A_mat[1, 2], A_mat[2, 2], A_mat[6, 2], A_mat[9, 2],
  #   A_mat[1, 6], A_mat[2, 6], A_mat[6, 6], A_mat[9, 6],
  #   A_mat[1, 9], A_mat[2, 9], A_mat[6, 9], A_mat[9, 9]
  # ))
  return SMatrix{4, 4, T, 16}((
    A_mat[1, 1], A_mat[9, 1], A_mat[6, 1], A_mat[2, 1],
    A_mat[1, 9], A_mat[9, 9], A_mat[6, 9], A_mat[2, 9],
    A_mat[1, 6], A_mat[9, 6], A_mat[6, 6], A_mat[2, 6],
    A_mat[1, 2], A_mat[9, 2], A_mat[6, 2], A_mat[2, 2],
  ))
end

#####################################################

# TODO better type this guy
struct Interpolants{A1, A2, A3, A4}
  X_q::A1
  N::A2
  ∇N_X::A3
  JxW::A4
end

#####################################################

struct NonAllocatedFunctionSpace{
  NDof,
  Conn    <: Connectivity,
  DofConn <: Connectivity,
  RefFE   <: ReferenceFE
} <: FunctionSpace{NDof, Conn, RefFE}

  conn::Conn
  dof_conn::DofConn
  ref_fe::RefFE
end

function Base.show(io::IO, fspace::NonAllocatedFunctionSpace)
  print(io::IO, "NonAllocatedFunctionSpace\n",
        "  Reference finite element = $(fspace.ref_fe)\n")
end

function NonAllocatedFunctionSpace(
  dof_manager::SimpleDofManager,
  conn::SimpleConnectivity, 
  q_degree::Int, 
  elem_type::Type{<:ReferenceFiniteElements.ReferenceFEType}
)

  ND       = num_dofs_per_node(dof_manager)
  NN, NE   = num_nodes_per_element(conn), num_elements(conn)
  ids      = dof_ids(dof_manager)
  # display(ids)
  temp     = reshape(ids[:, conn], ND * NN, NE)
  dof_conn = Connectivity{ND * NN, NE, Matrix, eltype(temp)}(temp)
  ref_fe   = ReferenceFE(elem_type(Val(q_degree)))
  return NonAllocatedFunctionSpace{ND, typeof(conn), typeof(dof_conn), typeof(ref_fe)}(
    conn, dof_conn, ref_fe
  )
end

function NonAllocatedFunctionSpace(
  dof_manager::VectorizedDofManager,
  conn::VectorizedConnectivity, 
  q_degree::Int, 
  elem_type::Type{<:ReferenceFiniteElements.ReferenceFEType}
)

  ND       = num_dofs_per_node(dof_manager)
  NN, NE   = num_nodes_per_element(conn), num_elements(conn)
  ids      = dof_ids(dof_manager)
  # display(ids)
  # temp     = reshape(ids[conn], ND * NN, NE)
  # temp     = ids[:, conn]
  ids      = reshape(1:ND * num_nodes(dof_manager), ND, num_nodes(dof_manager))
  temp     = reshape(ids[:, conn], ND * NN, NE)
  dof_conn = Connectivity{ND * NN, NE, Vector, eltype(temp)}(vec(temp))
  ref_fe   = ReferenceFE(elem_type(Val(q_degree)))
  return NonAllocatedFunctionSpace{ND, typeof(conn), typeof(dof_conn), typeof(ref_fe)}(
    conn, dof_conn, ref_fe
  )
end

NonAllocatedFunctionSpace(dof::DofManager, conn::Connectivity, q_degree::Int, elem_type::String) =
NonAllocatedFunctionSpace(dof, conn, q_degree, elem_type_map[elem_type])

# TODO try to move this to an abstract method
function Base.getindex(fspace::NonAllocatedFunctionSpace, X::NodalField, q::Int, e::Int)
  X_el = element_level_fields(fspace, X, e)
  N    = shape_function_values(fspace, q)
  ∇N_ξ = shape_function_gradients(fspace, q)
  ∇N_X = map_shape_function_gradients(X_el, ∇N_ξ)
  JxW  = volume(X_el, ∇N_ξ) * quadrature_weight(fspace, q)
  X_q  = X_el * N
  return Interpolants(X_q, N, ∇N_X, JxW)
end


########################################################

# FunctionSpace{}

struct VectorizedPreAllocatedFunctionSpace{
  NDof,
  Conn    <: Connectivity,
  DofConn <: Connectivity,
  RefFE   <: ReferenceFE,
  V1      <: QuadratureField,
  V2      <: QuadratureField,
  V3      <: QuadratureField
} <: FunctionSpace{NDof, Conn, RefFE}

  conn::Conn
  dof_conn::DofConn
  ref_fe::RefFE
  Ns::V1
  ∇N_Xs::V2
  JxWs::V3
end

function setup_shape_function_values!(Ns, ref_fe)
  for e in axes(Ns, 2)
    for q in axes(Ns, 1)
      Ns[q, e] = ReferenceFiniteElements.shape_function_values(ref_fe, q)
    end 
  end
end

function setup_shape_function_gradients!(∇N_Xs, Xs, conn, ref_fe)
  for e in axes(∇N_Xs, 2)
    X = Xs[e]
    for q in axes(∇N_Xs, 1)
      ∇N_ξ = ReferenceFiniteElements.shape_function_gradients(ref_fe, q)
      J     = X * ∇N_ξ
      J_inv = inv(J)
      ∇N_Xs[q, e]  = (J_inv * ∇N_ξ')'
    end
  end
end

function setup_shape_function_JxWs!(JxWs, Xs, ref_fe)
  for e in axes(JxWs, 2)
    X = Xs[e]
    for q in axes(JxWs, 1)
      ∇N_ξ = ReferenceFiniteElements.shape_function_gradients(ref_fe, q)
      J     = X * ∇N_ξ
      JxWs[q, e] = det(J) * ReferenceFiniteElements.quadrature_weight(ref_fe, q)
    end
  end
end

function VectorizedPreAllocatedFunctionSpace(
  dof_manager::VectorizedDofManager,
  conn::VectorizedConnectivity, 
  q_degree::Int, 
  elem_type::Type{<:ReferenceFiniteElements.ReferenceFEType},
  coords::VectorizedNodalField
)
  
  ND       = num_dofs_per_node(dof_manager)
  NN, NE   = num_nodes_per_element(conn), num_elements(conn)
  ids      = dof_ids(dof_manager)
  ids      = reshape(1:ND * num_nodes(dof_manager), ND, num_nodes(dof_manager))
  temp     = reshape(ids[:, conn], ND * NN, NE)
  dof_conn = Connectivity{ND * NN, NE, Vector, eltype(temp)}(vec(temp))
  ref_fe   = ReferenceFE(elem_type(q_degree))
  D        = ReferenceFiniteElements.num_dimensions(ref_fe)
  NQ       = ReferenceFiniteElements.num_q_points(ref_fe)
  Ns       = QuadratureField{NN, NQ, NE, Vector, SVector{NN, Float64}}(undef)
  ∇N_Xs    = QuadratureField{NN * D, NQ, NE, Vector, SMatrix{NN, D, Float64, NN * D}}(undef)
  JxWs     = QuadratureField{1, NQ, NE, Vector, Float64}(undef)
  Xs       = reinterpret(SMatrix{D, NN, eltype(coords), D * NN}, @views vec(coords[:, conn])) |> collect

  setup_shape_function_values!(Ns, ref_fe)
  setup_shape_function_gradients!(∇N_Xs, Xs, conn, ref_fe)
  setup_shape_function_JxWs!(JxWs, Xs, ref_fe)

  return VectorizedPreAllocatedFunctionSpace{
    ND, typeof(conn), typeof(dof_conn), typeof(ref_fe), typeof(Ns), typeof(∇N_Xs), typeof(JxWs)
  }(conn, dof_conn, ref_fe, Ns, ∇N_Xs, JxWs)
end