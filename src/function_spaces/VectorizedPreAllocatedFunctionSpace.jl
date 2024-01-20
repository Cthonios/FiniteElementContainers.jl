"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct VectorizedPreAllocatedFunctionSpace{
  NDof,
  # Conn    <: Connectivity,
  Conn    <: AbstractArray,
  # DofConn <: Connectivity,
  RefFE   <: ReferenceFE,
  V1      <: QuadratureField,
  V2      <: QuadratureField,
  V3      <: QuadratureField
} <: FunctionSpace{NDof, Conn, RefFE}

  conn::Conn
  # dof_conn::DofConn
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
    # X = Xs[e]
    X = Xs[:, conn[e]]
    for q in axes(∇N_Xs, 1)
      ∇N_ξ = ReferenceFiniteElements.shape_function_gradients(ref_fe, q)
      J     = (X * ∇N_ξ)'
      J_inv = inv(J)
      ∇N_Xs[q, e]  = (J_inv * ∇N_ξ')'
    end
  end
end

function setup_shape_function_JxWs!(JxWs, Xs, conn, ref_fe)
  for e in axes(JxWs, 2)
    # X = Xs[e]
    X = Xs[:, conn[e]]
    for q in axes(JxWs, 1)
      ∇N_ξ = ReferenceFiniteElements.shape_function_gradients(ref_fe, q)
      J     = (X * ∇N_ξ)'
      JxWs[q, e] = det(J) * ReferenceFiniteElements.quadrature_weights(ref_fe, q)
    end
  end
end

function VectorizedPreAllocatedFunctionSpace(
  dof_manager::DofManager,
  # conn::VectorizedConnectivity, 
  conn,
  q_degree::Int, 
  elem_type::Type{<:ReferenceFiniteElements.ReferenceFEType},
  coords::VectorizedNodalField
)
  
  ND       = num_dofs_per_node(dof_manager)
  # NN, NE   = num_nodes_per_element(conn), num_elements(conn)
  # NN       = num_nodes_per_element(elem_type), size(conn, 2)
  # NN, NE   = size(conn)
  NN, NE   = length(conn[1]), length(conn)
  ids      = dof_ids(dof_manager)
  ids      = reshape(1:ND * num_nodes(dof_manager), ND, num_nodes(dof_manager))
  # temp     = reshape(ids[:, conn], ND * NN, NE)
  # dof_conn = Connectivity{ND * NN, NE, Vector, eltype(temp)}(vec(temp))
  ref_fe   = ReferenceFE(elem_type(q_degree))
  D        = ReferenceFiniteElements.num_dimensions(ref_fe)
  NQ       = ReferenceFiniteElements.num_q_points(ref_fe)
  Ns       = QuadratureField{NN, NQ, NE, Vector, SVector{NN, Float64}}(undef)
  ∇N_Xs    = QuadratureField{NN * D, NQ, NE, Vector, SMatrix{NN, D, Float64, NN * D}}(undef)
  JxWs     = QuadratureField{1, NQ, NE, Vector, Float64}(undef)
  # Xs       = reinterpret(SMatrix{D, NN, eltype(coords), D * NN}, @views vec(coords[:, conn])) |> collect
  # Xs       = reinterpret(SMatrix{D, NN, eltype(coords), D * NN}, @views vec(coords[conn]))

  setup_shape_function_values!(Ns, ref_fe)
  setup_shape_function_gradients!(∇N_Xs, coords, conn, ref_fe)
  setup_shape_function_JxWs!(JxWs, coords, conn, ref_fe)

  return VectorizedPreAllocatedFunctionSpace{
    ND, typeof(conn), typeof(ref_fe), typeof(Ns), typeof(∇N_Xs), typeof(JxWs)
  }(conn, ref_fe, Ns, ∇N_Xs, JxWs)
end

VectorizedPreAllocatedFunctionSpace(dof::DofManager, conn, q_degree::Int, elem_type::String, coords) =
VectorizedPreAllocatedFunctionSpace(dof, conn, q_degree, elem_type_map[elem_type], coords)
