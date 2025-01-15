"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct VectorizedPreAllocatedFunctionSpace{
  NDof,
  Map,
  Conn    <: Connectivity,
  DofConn <: Connectivity,
  RefFE   <: ReferenceFE,
  V1      <: QuadratureField,
  V2      <: QuadratureField,
  V3      <: QuadratureField
} <: FunctionSpace{NDof, Conn, RefFE}
  elem_id_map::Map
  conn::Conn
  dof_conn::DofConn
  ref_fe::RefFE
  Ns::V1
  ∇N_Xs::V2
  JxWs::V3
end

function Base.show(io::IO, fspace::VectorizedPreAllocatedFunctionSpace)
  print(io::IO, "VectorizedPreAllocatedFunctionSpace\n",
        "  Reference finite element = $(fspace.ref_fe)\n")
end

function setup_shape_function_values!(Ns, ref_fe)
  for e in axes(Ns, 2)
    for q in axes(Ns, 1)
      Ns[q, e] = ReferenceFiniteElements.shape_function_value(ref_fe, q)
    end 
  end
end

# TODO check this
function setup_shape_function_gradients!(∇N_Xs, Xs, conn, ref_fe)
  for e in axes(∇N_Xs, 2)
    X = Xs[:, conn[:, e]] # TODO clean this up
    for q in axes(∇N_Xs, 1)
      ∇N_ξ = ReferenceFiniteElements.shape_function_gradient(ref_fe, q)
      J     = (X * ∇N_ξ)'
      J_inv = inv(J)
      ∇N_Xs[q, e]  = (J_inv * ∇N_ξ')'
    end
  end
end

# TODO check this
function setup_shape_function_JxWs!(JxWs, Xs, conn, ref_fe)
  for e in axes(JxWs, 2)
    X = Xs[:, conn[:, e]] # TODO clean this up
    for q in axes(JxWs, 1)
      ∇N_ξ = ReferenceFiniteElements.shape_function_gradient(ref_fe, q)
      J     = (X * ∇N_ξ)'
      JxWs[q, e] = det(J) * ReferenceFiniteElements.quadrature_weight(ref_fe, q)
    end
  end
end

function VectorizedPreAllocatedFunctionSpace(
  dof_manager::DofManager,
  elem_id_map,
  conn,
  q_degree::Int, 
  elem_type::Type{<:ReferenceFiniteElements.AbstractElementType},
  # coords::VectorizedNodalField
  coords::NodalField
)
  
  ND       = num_dofs_per_node(dof_manager)
  NN, NE   = num_nodes_per_element(conn), num_elements(conn)
  ids      = reshape(dof_ids(dof_manager), ND, size(dof_manager, 2))
  temp     = reshape(ids[:, conn], ND * NN, NE)
  dof_conn = Connectivity{ND * NN, NE, Matrix, eltype(temp)}(temp)
  # ref_fe   = ReferenceFE(elem_type(q_degree))
  ref_fe   = ReferenceFE(elem_type{Lagrange, q_degree}())
  D        = ReferenceFiniteElements.dimension(ref_fe)
  NQ       = ReferenceFiniteElements.num_quadrature_points(ref_fe)
  Ns       = QuadratureField{NN, NQ, NE, Vector, SVector{NN, Float64}}(undef)
  ∇N_Xs    = QuadratureField{NN * D, NQ, NE, Vector, SMatrix{NN, D, Float64, NN * D}}(undef)
  JxWs     = QuadratureField{1, NQ, NE, Vector, Float64}(undef)

  setup_shape_function_values!(Ns, ref_fe)
  setup_shape_function_gradients!(∇N_Xs, coords, conn, ref_fe)
  setup_shape_function_JxWs!(JxWs, coords, conn, ref_fe)

  return VectorizedPreAllocatedFunctionSpace{
    ND, typeof(elem_id_map), typeof(conn), typeof(dof_conn), typeof(ref_fe), typeof(Ns), typeof(∇N_Xs), typeof(JxWs)
  }(elem_id_map, conn, dof_conn, ref_fe, Ns, ∇N_Xs, JxWs)
end

VectorizedPreAllocatedFunctionSpace(dof::DofManager, elem_id_map, conn, q_degree::Int, elem_type::String, coords) =
VectorizedPreAllocatedFunctionSpace(dof, elem_id_map, conn, q_degree, elem_type_map[elem_type], coords)

# TODO try to move this to an abstract method
function Base.getindex(fspace::VectorizedPreAllocatedFunctionSpace, X::NodalField, q::Int, e::Int)
  X_el = element_level_coordinates(fspace, X, e)
  N    = fspace.Ns[q, e]
  ∇N_X = fspace.∇N_Xs[q, e]
  JxW  = fspace.JxWs[q, e]
  X_q  = X_el * N
  return Interpolants(X_q, N, ∇N_X, JxW)
end
