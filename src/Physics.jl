abstract type AbstractPhysics{NF, NP, NS} end
num_fields(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS} = NF
num_properties(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS} = NP
num_states(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS} = NS

# physics like methods
function damping end
function energy end
function mass end
function mass! end
function mass_action end
function mass_action! end
function residual end
function residual! end
function stiffness end
function stiffness! end
function stiffness_action end
function stiffness_action! end

# optimization like methods
function gradient end
function hessian end
function value end

# default
function create_function(fspace, physics::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS}
  names = field_names(physics)
  return GeneralFunction(fspace, names)
end

# default
function create_initial_state(::AbstractPhysics{NF, NP, 0}) where {NF, NP}
  return SVector{0, Float64}()
end

"""
$(TYPEDSIGNATURES)
"""
function create_properties(physics::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS}
  @assert false "You need to implement the create_properties method for physics $(physics) or 
  type $(typeof(physics))!"
end

# default
function create_properties(::AbstractPhysics{NF, 0, NS}) where {NF, NS}
  return SVector{0, Float64}()
end

# default
function field_names(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS}
  if NF == 0
    field_names = String[]
  else
    field_names = map(x -> "field_$x", 1:NF)
  end
  return field_names
end

@inline function interpolate_field_values(
  physics::P, interps::M, U_el::SVector{N, T}
) where {P <: AbstractPhysics, M <: MappedH1OrL2Interpolants, N, T}
  U_el = reshape_element_level_field(physics, U_el)
  return U_el * interps.N
end

@inline function interpolate_field_gradients(
  physics::P, interps::M, U_el::SVector{N, T}
) where {P <: AbstractPhysics, M <: MappedH1OrL2Interpolants, N, T}
  # @show size(U_el)
  U_el = reshape_element_level_field(physics, U_el)
  # return U_el * interps.∇N_X
  # @show size(U_el)
  # @show size(interps.∇N_X)
  # return (U_el * interps.∇N_X')'
  return U_el * interps.∇N_X
end

@inline function interpolate_field_values_and_gradients(
  physics::P, interps::M, U_el::SVector{N, T}
) where {P <: AbstractPhysics, M <: MappedH1OrL2Interpolants, N, T}
  U_el = reshape_element_level_field(physics, U_el)
  # return U_el * interps.N, (U_el * interps.∇N_X')'
  return U_el * interps.N, U_el * interps.∇N_X
end

@inline function map_interpolants(
  interps::I, x_el::SVector{NxD, T}
) where {I <: ReferenceFiniteElements.AbstractInterpolants, NxD, T <: Number}
  x_el = reshape_element_level_coordinates(interps, x_el)
  interps = MappedH1OrL2Interpolants(interps, x_el)
  return interps
end

# TODO to make bcs a little clenaer
# @inline function map_surface_interpolants(
#   interps::I, x_el::SVector{NxD, T}
# ) where {I <: ReferenceFiniteElements.AbstractInterpolants, NxD, T <: Number}
#   x_el = reshape_element_level_coordinates(interps, x_el)
#   interps = MappedH1OrL2Interpolants(interps, x_el)
#   return interps
# end

# default
function property_names(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS}
  if NP == 0
    prop_names = String[]
  else
    prop_names = map(x -> "property_$x", 1:NP)
  end
  return prop_names
end

"""
$(TYPEDSIGNATURES)
"""
@inline function reshape_element_level_coordinates(
  interps::I, x_el::SVector{NxD, T}
) where {I <: ReferenceFiniteElements.AbstractInterpolants, NxD, T <: Number}
  ND = dimension(interps)
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

# default
function state_variable_names(::AbstractPhysics{NF, NP, NS}) where {NF, NP, NS}
  if NS == 0
    state_var_names = String[]
  else
    state_var_names = map(x -> "state_variable_$x", 1:NS)
  end
  return state_var_names
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
