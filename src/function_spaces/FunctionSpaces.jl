include("Utils.jl")

#####################################################

"""
$(TYPEDEF)
"""
abstract type AbstractFunctionSpace{NDof, RefFE, Conn} end
"""
$(TYPEDSIGNATURES)
"""
connectivity(fspace::AbstractFunctionSpace)             = fspace.conn
"""
$(TYPEDSIGNATURES)
"""
connectivity(fspace::AbstractFunctionSpace, e::Int)     = connectivity(fspace.conn, e)
"""
$(TYPEDSIGNATURES)
"""
dof_connectivity(fspace::AbstractFunctionSpace)         = fspace.dof_conn
"""
$(TYPEDSIGNATURES)
"""
dof_connectivity(fspace::AbstractFunctionSpace, e::Int) = connectivity(fspace.dof_conn, e)
"""
$(TYPEDSIGNATURES)
"""
reference_element(fspace::AbstractFunctionSpace)        = fspace.ref_fe
"""
$(TYPEDSIGNATURES)
"""
num_dimensions(fspace::AbstractFunctionSpace)           = ReferenceFiniteElements.dimension(fspace.ref_fe.element)
"""
$(TYPEDSIGNATURES)
"""
num_elements(fspace::AbstractFunctionSpace)             = num_elements(fspace.conn)
"""
$(TYPEDSIGNATURES)
"""
num_nodes_per_element(fspace::AbstractFunctionSpace)    = ReferenceFiniteElements.num_vertices(fspace.ref_fe)
"""
$(TYPEDSIGNATURES)
"""
num_q_points(fspace::AbstractFunctionSpace)             = ReferenceFiniteElements.num_quadrature_points(fspace.ref_fe)
"""
$(TYPEDSIGNATURES)
"""
num_dofs_per_node(::AbstractFunctionSpace{ND, RefFE, Conn}) where {ND, RefFE, Conn} = ND

#####################################################
"""
$(TYPEDSIGNATURES)
"""
quadrature_points(fspace::AbstractFunctionSpace, q::Int) =
ReferenceFiniteElements.quadrature_point(fspace.ref_fe, q)
"""
$(TYPEDSIGNATURES)
"""
quadrature_weights(fspace::AbstractFunctionSpace, q::Int) = 
ReferenceFiniteElements.quadrature_weight(fspace.ref_fe, q)
"""
$(TYPEDSIGNATURES)
"""
shape_function_values(fspace::AbstractFunctionSpace, q::Int) = 
ReferenceFiniteElements.shape_function_value(fspace.ref_fe, q) 
"""
$(TYPEDSIGNATURES)
"""
shape_function_gradients(fspace::AbstractFunctionSpace, q::Int) = 
ReferenceFiniteElements.shape_function_gradient(fspace.ref_fe, q)
"""
$(TYPEDSIGNATURES)
"""
shape_function_hessians(fspace::AbstractFunctionSpace, q::Int) = 
ReferenceFiniteElements.shape_function_hessian(fspace.ref_fe, q)

#####################################################
"""
$(TYPEDSIGNATURES)
"""
function element_level_coordinates(fspace::AbstractFunctionSpace, x, e::Int)
  NF, NN = num_fields(x), num_nodes_per_element(fspace)
  u_el = SMatrix{NF, NN, eltype(x), NF * NN}(@views vec(x[:, connectivity(fspace, e)]))
  return u_el
end
"""
$(TYPEDSIGNATURES)
"""
function element_level_fields(fspace::AbstractFunctionSpace, u::NodalField)
  u_el = element_level_fields_reinterpret(fspace, u)
  NN, NE = num_nodes_per_element(fspace), num_elements(fspace)
  ND = num_dofs_per_node(fspace)
  u_el_ret = ElementField{(NN * ND, NE), eltype(u_el)}(undef)
  u_el_ret .= u_el
end
"""
$(TYPEDSIGNATURES)
"""
function element_level_fields(fspace::AbstractFunctionSpace, u, e::Int)
  NF, NN = num_fields(u), num_nodes_per_element(fspace)
  u_el = SMatrix{NF, NN, eltype(u), NF * NN}(@views vec(u[:, connectivity(fspace, e)]))
  return u_el
end
"""
$(TYPEDSIGNATURES)
"""
function element_level_fields_reinterpret(fspace::AbstractFunctionSpace, u::NodalField)
  NF, NN = num_fields(u), num_nodes_per_element(fspace)
  u_el = reinterpret(SMatrix{NF, NN, eltype(u), NF * NN}, @views vec(u[:, connectivity(fspace)]))
  return u_el
end
"""
$(TYPEDSIGNATURES)
"""
function element_level_fields_reinterpret(fspace::AbstractFunctionSpace, u::NodalField, e::Int)
  NF, NN = num_fields(u), num_nodes_per_element(fspace)
  u_el = reinterpret(SMatrix{NF, NN, eltype(u), NF * NN}, @views vec(u[:, connectivity(fspace, e)]))
  return u_el
end
"""
$(TYPEDSIGNATURES)
"""
function quadrature_level_field_values(fspace::AbstractFunctionSpace, ::NodalField, u::NodalField, q::Int, e::Int)
  u_el = element_level_fields(fspace, u, e)
  N_q  = shape_function_values(fspace, q)
  return u_el * N_q
end
"""
$(TYPEDSIGNATURES)
"""
function quadrature_level_field_gradients(fspace::AbstractFunctionSpace, X::NodalField, u::NodalField, q::Int, e::Int)
  X_el = element_level_coordinates(fspace, X, e)
  u_el = element_level_fields(fspace, u, e)
  ∇N_ξ = shape_function_gradients(fspace, q)
  ∇N_X = map_shape_function_gradients(X_el, ∇N_ξ)
  return u_el * ∇N_X
end

# all ```volume``` methods below are incorrect
# use shape_function_gradient_and_volume instead
"""
$(TYPEDSIGNATURES)
"""
volume(fspace::AbstractFunctionSpace, ::ReferenceFiniteElements.AbstractElementType, X::NodalField, q::Int, e::Int) =
fspace[X, q, e].JxW
# volume(fspace::AbstractFunctionSpace, X::NodalField, q::Int, e::Int) = fspace[X, q, e].JxW

"""
$(TYPEDSIGNATURES)
"""
function volume(fspace::AbstractFunctionSpace, X::NodalField, e::Int)
  v = 0.0 # TODO place for unitful to not work
  for q in 1:num_q_points(fspace)
    v = v + volume(fspace, fspace.ref_fe.element, X, q, e)
  end
  return v
end
"""
$(TYPEDSIGNATURES)
"""
function volume(fspace::AbstractFunctionSpace, X::NodalField)
  v = 0.0
  for e in 1:num_elements(fspace)
    for q in 1:num_q_points(fspace)
      v = v + volume(fspace, fspace.ref_fe.element, X, q, e)
    end
  end
  return v
end

function shape_function_gradient_and_volume(ref_fe::ReferenceFE, X_el, q::Int)
  ∇N_ξ = ReferenceFiniteElements.shape_function_gradient(ref_fe, q)
  w    = ReferenceFiniteElements.quadrature_weight(ref_fe, q)
  # return shape_function_gradient_and_volume(ref_fe.ref_fe_type, X_el, ∇N_ξ, w)
  J = (X_el * ∇N_ξ)'
  J_inv = inv(J)
  ∇N_X = (J_inv * ∇N_ξ')'
  JxW = det(J) * w
  return ∇N_X, JxW
end

##############################################################################

# TODO better type this guy
"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct Interpolants{A1, A2, A3, A4}
  X_q::A1
  N::A2
  ∇N_X::A3
  JxW::A4
end

# Implementations
include("NonAllocatedFunctionSpace.jl")
include("VectorizedPreAllocatedFunctionSpace.jl")
