abstract type AbstractPhysics{NF, NP, NS} end

num_fields(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS} = NF
num_properties(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS} = NP
num_states(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS} = NS

"""
$(TYPEDSIGNATURES)
"""
function create_properties(physics::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS}
  @assert false "You need to implement the create_properties method for physics $(physics) or 
  type $(typeof(physics))!"
end

function create_properties(::AbstractPhysics{NF, 0, NS}) where {NF, NS}
  return SVector{0, Float64}()
end

function create_initial_state(::AbstractPhysics{NF, NP, 0}) where {NF, NP}
  return SVector{0, Float64}()
end

@inline function interpolate_field_values(
  physics::P, interps::M, U_el::SVector{N, T}
) where {P <: AbstractPhysics, M <: MappedInterpolants, N, T}
  U_el = reshape_element_level_field(physics, U_el)
  return U_el * interps.N
end

@inline function interpolate_field_gradients(
  physics::P, interps::M, U_el::SVector{N, T}
) where {P <: AbstractPhysics, M <: MappedInterpolants, N, T}
  U_el = reshape_element_level_field(physics, U_el)
  return U_el * interps.∇N_X
end

@inline function interpolate_field_values_and_gradients(
  physics::P, interps::M, U_el::SVector{N, T}
) where {P <: AbstractPhysics, M <: MappedInterpolants, N, T}
  U_el = reshape_element_level_field(physics, U_el)
  return U_el * interps.N, U_el * interps.∇N_X
end

@inline function map_interpolants(
  interps::I, x_el::SVector{NxD, T}
) where {I <: ReferenceFiniteElements.AbstractInterpolants, NxD, T <: Number}
  x_el = reshape_element_level_coordinates(interps, x_el)
  interps = MappedInterpolants(interps, x_el)
  return interps
end

"""
$(TYPEDSIGNATURES)
"""
@inline function reshape_element_level_coordinates(
  interps::I, x_el::SVector{NxD, T}
) where {I <: ReferenceFiniteElements.AbstractInterpolants, NxD, T <: Number}
  # ND = ReferenceFiniteElements.num_dimensions(interps)
  ND = size(interps.∇N_ξ, 2)
  N = NxD ÷ ND
  return SMatrix{ND, N, T, NxD}(x_el)
end

"""
$(TYPEDSIGNATURES)
"""
@inline function reshape_element_level_field(
  physics::P, u_el::SVector{NxNDof, T}
) where {P <: AbstractPhysics, NxNDof, T <: Number}
  NDof = num_fields(physics)
  N = NxNDof ÷ NDof
  return SMatrix{NDof, N, T, NxNDof}(u_el)
end

"""
$(TYPEDSIGNATURES)
Unpacks a single fields values from a SMatrix.
This is useful for extracting specific components
from an interpolated field gradient at a
quadrature point ∇u_q.

Return a SVector
"""
@inline function unpack_field(field::SMatrix{M, N, T, L}, dof::Int) where {M, N, T <: Number, L}
  ids = SVector{N, Int}(1:N)
  return SVector{N, T}(field[dof, ids])
end

"""
$(TYPEDSIGNATURES)
Unpacks a range of fields values from a SMatrix.
This is useful for extracting specific components
from an interpolated field gradient at a
quadrature point ∇u_q.

returns a SMatrix.

Note the Val{D} that is a necessary input. This is
crucial for performance with ```StaticArrays```.
"""
@inline function unpack_field(
  field::SMatrix{M, N, T, L}, dof_start::Int, dof_end::Int,
  ::Val{D}
) where {M, N, T <: Number, L, D}
  @assert D == dof_end - dof_start + 1 "Number of expected dofs wrapped in Val is not equal to dof_end - dof_start"
  ids = SVector{N, Int}(1:N)
  dofs = SVector{D, Int}(dof_start:dof_end)
  return SMatrix{D, N, T, D * N}(field[dofs, ids])
end

# Can we make something that makes interfacing with kernels easier?
# How can we make something like this work nicely with AD?
# struct PhysicsQuadratureState{T, ND, NN, NF, NP, NS, NDxNF, NNxND, NNxNNxND}
#   # u::SVector{NF, T}
#   # ∇u::SMatrix{NF, ND, T, NDxNF}
#   # props::SVector{NP, T}
#   # state_old::SVector{NS, T}
#   # interpolants and gauss weight at quadrature point
#   N::SVector{NN, T}
#   ∇N_ξ::SMatrix{NN, ND, T, NNxND}
#   ∇∇N_ξ::SArray{Tuple{NN, ND, ND}, T, 3, NNxNNxND}
#   JxW::T
#   # element level fields
#   u_el::SMatrix{}
# end 

# physics like methods
function damping end
function energy end
function mass end
function residual end
function stiffness end

# optimization like methods
function gradient end
function hessian end
function value end

# function energy(::AbstractPhysics)

# end

# function residual(
#   physics, interps::ReferenceFiniteElements.Interpolants,
#   u_el, x_el, state_old_q, props_el, dt
# )
#   mapped_interps = MappedInterpolants(interps, x_el)
#   return residual(physics, mapped_interps, u_el, state_old_q, props_el, dt)
# end
