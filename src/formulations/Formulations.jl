"""
$(TYPEDEF)
"""
abstract type AbstractMechanicsFormulation{ND} end

"""
$(TYPEDSIGNATURES)
"""
num_dimensions(::AbstractMechanicsFormulation{ND}) where ND = ND

# for large tuples, otherwise we get allocations
@generated function set(t::Tuple{Vararg{Any, N}}, x, i) where {N}
  Expr(:tuple, (:(ifelse($j == i, x, t[$j])) for j in 1:N)...)
end

function extract_stiffness end
function extract_stress end

include("IncompressiblePlaneStress.jl")
include("PlaneStrain.jl")
include("ScalarFormulation.jl")
include("ThreeDimensional.jl")

"""
$(TYPEDSIGNATURES)
"""
function discrete_gradient(fspace::AbstractFunctionSpace, type::T, X, q, e) where T <: AbstractMechanicsFormulation
  X_el = element_level_fields(fspace, X, e)
  ∇N_X = map_shape_function_gradients(X_el, shape_function_gradients(fspace, q))
  G    = discrete_gradient(type, ∇N_X)
  return G
end

"""
$(TYPEDSIGNATURES)
"""
function discrete_symmetric_gradient(fspace, type::T, X, q, e) where T <: AbstractMechanicsFormulation
  X_el = element_level_fields(fspace, X, e)
  ∇N_X = map_shape_function_gradients(X_el, shape_function_gradients(fspace, q))
  G    = discrete_symmetric_gradient(type, ∇N_X)
  return G
end
