abstract type AbstractParameters end

struct Parameters{S, P, H1 <: H1Field} <: AbstractParameters
  # actual parameter fields
  state_old::S
  state_new::S
  properties::P
  # scratch fields
  h1_field::H1
end

function Parameters(dof::DofManager, physics::Vector{<:AbstractPhysics})
  n_state = num_states(physics)
end