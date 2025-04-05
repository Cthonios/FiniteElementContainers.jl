abstract type AbstractParameters end

struct Parameters{P, S, H1} <: AbstractParameters
  # actual parameter fields
  # TODO add boundary condition stuff and time stepping stuff
  properties::P
  state_old::S
  state_new::S
  # scratch fields
  h1_field::H1
end

# TODO only works for H1Fields currently most likely
function Parameters(dof::DofManager, physics::AbstractPhysics)
  n_props = num_properties(physics)
  n_state = num_states(physics)

  # TODO
  fspace = dof.H1_vars[1].fspace

  for (name, ref_fe) in pairs(fspace.ref_fes)

  end
end
