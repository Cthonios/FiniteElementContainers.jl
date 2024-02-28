"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
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
  dof_manager::DofManager,
  conn::SimpleConnectivity, 
  q_degree::Int, 
  elem_type::Type{<:ReferenceFiniteElements.ReferenceFEType}
)

  ND       = num_dofs_per_node(dof_manager)
  NN, NE   = num_nodes_per_element(conn), num_elements(conn)
  ids      = reshape(dof_ids(dof_manager), ND, size(dof_manager, 2))
  # display(ids)
  temp     = reshape(ids[:, conn], ND * NN, NE)
  dof_conn = Connectivity{ND * NN, NE, Matrix, eltype(temp)}(temp)
  ref_fe   = ReferenceFE(elem_type(Val(q_degree)))
  return NonAllocatedFunctionSpace{ND, typeof(conn), typeof(dof_conn), typeof(ref_fe)}(
    conn, dof_conn, ref_fe
  )
end

function NonAllocatedFunctionSpace(
  dof_manager::DofManager,
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

  # new addition, may not be general enough
  # conn = Connectivity{NN, NE, Vector, SVector}(conn.vals)
  # dof_conn = Connectivity{ND * NN, NE, Vector, SVector}(dof_conn.vals)

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
  JxW  = volume(X_el, ∇N_ξ) * quadrature_weights(fspace, q)
  X_q  = X_el * N
  return Interpolants(X_q, N, ∇N_X, JxW)
end
