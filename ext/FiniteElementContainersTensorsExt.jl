module FiniteElementContainersTensorsExt

using DocStringExtensions
using FiniteElementContainers
using StaticArrays
using Tensors

# IncompressiblePlaneStress

"""
$(TYPEDSIGNATURES)
"""
function FiniteElementContainers.modify_field_gradients(::IncompressiblePlaneStress, ∇u_q::SMatrix{2, 2, T, 4}, ::Type{<:Tensor}) where T <: Number
  return Tensor{2, 3, T, 9}((
    ∇u_q[1, 1], ∇u_q[2, 1], 0.0,
    ∇u_q[1, 2], ∇u_q[2, 2], 0.0,
    0.0,        0.0,        1.0 / det(∇u_q)
  ))
end

"""
$(TYPEDSIGNATURES)
"""
function FiniteElementContainers.modify_field_gradients(::IncompressiblePlaneStress, ∇u_q::Tensor{2, 2, T, 4}, ::Type{<:Tensor}) where T <: Number
  return Tensor{2, 3, T, 9}((
    ∇u_q[1, 1], ∇u_q[2, 1], 0.0,
    ∇u_q[1, 2], ∇u_q[2, 2], 0.0,
    0.0,        0.0,        1.0 / det(∇u_q)
  ))
end

"""
$(TYPEDSIGNATURES)
"""
function FiniteElementContainers.extract_stress(::IncompressiblePlaneStress, P::Tensor{2, 3, T, 9}) where T <: Number
  P_vec = tovoigt(SVector, P)
  return SVector{4, T}((
    P_vec[1], P_vec[9], 
    P_vec[6], P_vec[2]
  ))
end
"""
$(TYPEDSIGNATURES)
"""
function FiniteElementContainers.extract_stiffness(::IncompressiblePlaneStress, A::Tensor{4, 3, T, 81}) where T <: Number
  A_mat = tovoigt(SMatrix, A)
  return SMatrix{4, 4, T, 16}((
    A_mat[1, 1], A_mat[9, 1], A_mat[6, 1], A_mat[2, 1],
    A_mat[1, 9], A_mat[9, 9], A_mat[6, 9], A_mat[2, 9],
    A_mat[1, 6], A_mat[9, 6], A_mat[6, 6], A_mat[2, 6],
    A_mat[1, 2], A_mat[9, 2], A_mat[6, 2], A_mat[2, 2],
  ))
end

# PlaneStrain

"""
$(TYPEDSIGNATURES)
"""
function FiniteElementContainers.modify_field_gradients(::PlaneStrain, ∇u_q::SMatrix{2, 2, T, 4}, ::Type{<:Tensor}) where T <: Number
  return Tensor{2, 3, T, 9}((
    ∇u_q[1, 1], ∇u_q[2, 1], 0.0,
    ∇u_q[1, 2], ∇u_q[2, 2], 0.0,
    0.0,        0.0,        0.0
  ))
end

"""
To deprecate or not to deprecate?
$(TYPEDSIGNATURES)
"""
FiniteElementContainers.modify_field_gradients(form::PlaneStrain, ∇u_q::SMatrix{2, 2, T, 4}; type = Tensor) where T <: Number =
modify_field_gradients(form, ∇u_q, type)

"""
$(TYPEDSIGNATURES)
"""
function FiniteElementContainers.modify_field_gradients(::PlaneStrain, ∇u_q::Tensor{2, 2, T, 4}, ::Type{<:Tensor}) where T <: Number
  return Tensor{2, 3, T, 9}((
    ∇u_q[1, 1], ∇u_q[2, 1], 0.0,
    ∇u_q[1, 2], ∇u_q[2, 2], 0.0,
    0.0,        0.0,        0.0
  ))
end

"""
$(TYPEDSIGNATURES)
"""
function FiniteElementContainers.extract_stress(::PlaneStrain, P::Tensor{2, 3, T, 9}) where T <: Number
  P_vec = tovoigt(SVector, P)
  return SVector{4, T}((
    P_vec[1], P_vec[9], 
    P_vec[6], P_vec[2]
  ))
end
"""
$(TYPEDSIGNATURES)
"""
function FiniteElementContainers.extract_stiffness(::PlaneStrain, A::Tensor{4, 3, T, 81}) where T <: Number
  A_mat = tovoigt(SMatrix, A)
  return SMatrix{4, 4, T, 16}((
    A_mat[1, 1], A_mat[9, 1], A_mat[6, 1], A_mat[2, 1],
    A_mat[1, 9], A_mat[9, 9], A_mat[6, 9], A_mat[2, 9],
    A_mat[1, 6], A_mat[9, 6], A_mat[6, 6], A_mat[2, 6],
    A_mat[1, 2], A_mat[9, 2], A_mat[6, 2], A_mat[2, 2],
  ))
end

# ThreeDimensional

"""
$(TYPEDSIGNATURES)
"""
function FiniteElementContainers.modify_field_gradients(::ThreeDimensional, ∇u_q, ::Type{<:Tensor})
  return Tensor{2, 3, eltype(∇u_q), 9}(∇u_q)
end

"""
$(TYPEDSIGNATURES)
"""
FiniteElementContainers.modify_field_gradients(form::ThreeDimensional, ∇u_q, type = Tensor) =
modify_field_gradients(form, ∇u_q, type)

"""
$(TYPEDSIGNATURES)
"""
function FiniteElementContainers.extract_stress(::ThreeDimensional, P::Tensor{2, 3, T, 9}) where T <: Number
  P_vec = tovoigt(SVector, P)
  return SVector{9, T}((
    P_vec[1], P_vec[9], P_vec[8],
    P_vec[6], P_vec[2], P_vec[7],
    P_vec[5], P_vec[4], P_vec[3]
  ))
end

"""
$(TYPEDSIGNATURES)
"""
function FiniteElementContainers.extract_stiffness(::ThreeDimensional, A_in::Tensor{4, 3, T, 81}) where T <: Number
  A = tovoigt(SMatrix, A_in)
  return SMatrix{9, 9, T, 81}((
    A[1, 1], A[9, 1], A[8, 1], A[6, 1], A[2, 1], A[7, 1], A[5, 1], A[4, 1], A[3, 1],
    A[1, 9], A[9, 9], A[8, 9], A[6, 9], A[2, 9], A[7, 9], A[5, 9], A[4, 9], A[3, 9],
    A[1, 8], A[9, 8], A[8, 8], A[6, 8], A[2, 8], A[7, 8], A[5, 8], A[4, 8], A[3, 8],
    A[1, 6], A[9, 6], A[8, 6], A[6, 6], A[2, 6], A[7, 6], A[5, 6], A[4, 6], A[3, 6],
    A[1, 2], A[9, 2], A[8, 2], A[6, 2], A[2, 2], A[7, 2], A[5, 2], A[4, 2], A[3, 2],
    A[1, 7], A[9, 7], A[8, 7], A[6, 7], A[2, 7], A[7, 7], A[5, 7], A[4, 7], A[3, 7],
    A[1, 5], A[9, 5], A[8, 5], A[6, 5], A[2, 5], A[7, 5], A[5, 5], A[4, 5], A[3, 5],
    A[1, 4], A[9, 4], A[8, 4], A[6, 4], A[2, 4], A[7, 4], A[5, 4], A[4, 4], A[3, 4],
    A[1, 3], A[9, 3], A[8, 3], A[6, 3], A[2, 3], A[7, 3], A[5, 3], A[4, 3], A[3, 3],
  ))
end

end # module
