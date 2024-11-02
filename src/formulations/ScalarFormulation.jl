"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct ScalarFormulation <: AbstractMechanicsFormulation{1}
end

"""
$(TYPEDSIGNATURES)
"""
function discrete_gradient(::ScalarFormulation, ∇N_X)
  return ∇N_X
end

"""
$(TYPEDSIGNATURES)
"""
function discrete_symmetric_gradient(::ScalarFormulation, ∇N_X)
  # TODO
end

"""
$(TYPEDSIGNATURES)
"""
function discrete_values(::ScalarFormulation, N)
  return N
end

function modify_field_gradients(::ScalarFormulation, ∇u_q)
  return ∇u_q
end
