module FiniteElementContainersKernelAbstractionsExt

using FiniteElementContainers
using KernelAbstractions

function FiniteElementContainers.create_fields(
  backend::Backend, 
  ::DofManager{T, N, ND, NN, B}, 
  float_type::Type{<:Number} = Float64
) where {T, N, ND, NN, B <: AbstractArray{Bool, 1}}
  vals = KernelAbstractions.zeros(backend, float_type, ND * NN)
  return NodalField{ND, NN, Vector}(vals)
end

function FiniteElementContainers.create_fields(
  backend::Backend, 
  ::DofManager{T, N, ND, NN, B}, 
  float_type::Type{<:Number} = Float64
) where {T, N, ND, NN, B <: AbstractArray{Bool, 2}}
  vals = KernelAbstractions.zeros(backend, float_type, ND, NN)
  return NodalField{ND, NN, Matrix}(vals)
end

end # module