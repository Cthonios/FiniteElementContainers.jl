"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct NonAllocatedFunctionSpace{
  NDof,
  Map,
  Conn    <: Connectivity,
  DofConn <: Connectivity,
  RefFE   <: ReferenceFE
} <: AbstractFunctionSpace{NDof, Conn, RefFE}
  elem_id_map::Map
  conn::Conn
  dof_conn::DofConn
  ref_fe::RefFE
end

function Base.show(io::IO, fspace::NonAllocatedFunctionSpace)
  print(io::IO, "NonAllocatedFunctionSpace\n",
        "  Reference finite element = $(fspace.ref_fe)\n")
end

# function NonAllocatedFunctionSpace(
#   dof_manager::DofManager,
#   elem_id_map,
#   conn::SimpleConnectivity, 
#   q_degree::Int, 
#   elem_type::Type{<:ReferenceFiniteElements.AbstractElementType}
# )

#   ND       = num_dofs_per_node(dof_manager)
#   NN, NE   = num_nodes_per_element(conn), num_elements(conn)
#   ids      = reshape(dof_ids(dof_manager), ND, size(dof_manager, 2))
#   temp     = reshape(ids[:, conn], ND * NN, NE)
#   dof_conn = Connectivity{ND * NN, NE, Matrix, eltype(temp)}(temp)
#   ref_fe   = ReferenceFE(elem_type{Lagrange, q_degree}())
#   return NonAllocatedFunctionSpace{ND, typeof(elem_id_map), typeof(conn), typeof(dof_conn), typeof(ref_fe)}(
#     elem_id_map, conn, dof_conn, ref_fe
#   )
# end

function NonAllocatedFunctionSpace(
  dof_manager::DofManager,
  elem_id_map,
  conn::Connectivity, 
  q_degree::Int, 
  elem_type::Type{<:ReferenceFiniteElements.AbstractElementType}
)

  ND       = num_dofs_per_node(dof_manager)
  NN, NE   = num_nodes_per_element(conn), num_elements(conn)
  ids      = dof_ids(dof_manager)
  ids      = reshape(1:ND * num_nodes(dof_manager), ND, num_nodes(dof_manager))
  temp     = reshape(ids[:, conn], ND * NN, NE)
  dof_conn = Connectivity{ND * NN, NE}(vec(temp))
  ref_fe   = ReferenceFE(elem_type{Lagrange, q_degree}())

  return NonAllocatedFunctionSpace{ND, typeof(elem_id_map), typeof(conn), typeof(dof_conn), typeof(ref_fe)}(
    elem_id_map, conn, dof_conn, ref_fe
  )
end

NonAllocatedFunctionSpace(dof::DofManager, elem_id_map, conn::Connectivity, q_degree::Int, elem_type::String) =
NonAllocatedFunctionSpace(dof, elem_id_map, conn, q_degree, elem_type_map[elem_type])

# TODO try to move this to an abstract method
function Base.getindex(fspace::NonAllocatedFunctionSpace, X::NodalField, q::Int, e::Int)
  X_el = element_level_coordinates(fspace, X, e)
  N    = shape_function_values(fspace, q)
  ∇N_ξ = shape_function_gradients(fspace, q)
  ∇N_X = map_shape_function_gradients(X_el, ∇N_ξ)
  JxW  = volume(X_el, ∇N_ξ) * quadrature_weights(fspace, q)
  X_q  = X_el * N
  return Interpolants(X_q, N, ∇N_X, JxW)
end
