abstract type AbstractParameters end

struct Parameters{D, N, Phys, Props, S, V, H1} <: AbstractParameters
  # actual parameter fields
  # TODO add boundary condition stuff and time stepping stuff
  dirichlet_bcs::D
  neumann_bcs::N
  physics::Phys
  #
  properties::Props
  state_old::S
  state_new::S
  # scratch fields
  h1_dbcs::V
  h1_field::H1
end

# TODO only works for H1Fields currently most likely
# function Parameters(dof::DofManager, physics::AbstractPhysics)
#   n_props = num_properties(physics)
#   n_state = num_states(physics)

#   # TODO
#   fspace = dof.H1_vars[1].fspace

#   for (name, ref_fe) in pairs(fspace.ref_fes)

#   end
# end

function Parameters(assembler, physics, dbcs, nbcs)
  h1_dbcs = create_bcs(assembler, H1Field)
  h1_field = create_field(assembler, H1Field)

  # TODO
  properties = nothing
  state_old = nothing
  state_new = nothing

  return Parameters(
    dbcs, nbcs, 
    physics, 
    properties, 
    state_old, state_new, 
    h1_dbcs, h1_field
  )
end

function create_parameters(assembler, physics, dbcs=nothing, nbcs=nothing)
  return Parameters(assembler, physics, dbcs, nbcs)
end

