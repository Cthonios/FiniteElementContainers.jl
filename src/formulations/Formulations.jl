"""
$(TYPEDSIGNATURES)
"""
abstract type AbstractMechanicsFormulation end

# for large tuples, otherwise we get allocations
@generated function set(t::Tuple{Vararg{Any, N}}, x, i) where {N}
  Expr(:tuple, (:(ifelse($j == i, x, t[$j])) for j in 1:N)...)
end

include("PlaneStrain.jl")
include("ThreeDimensional.jl")

"""
$(TYPEDSIGNATURES)
"""
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

"""
$(TYPEDSIGNATURES)
"""
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
