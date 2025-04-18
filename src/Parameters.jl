# move to a seperate file
abstract type AbstractTimeStepper{T} end
current_time(t::AbstractTimeStepper) = sum(t.time_current)
time_step(t::AbstractTimeStepper) = sum(t.Δt)

struct TimeStepper{T} <: AbstractTimeStepper{T}
  time_start::T
  time_end::T
  time_current::T
  Δt::T
end

function TimeStepper(time_start_in::T, time_end_in::T, n_steps::Int) where T <: Number
  time_start = zeros(1)
  time_end = zeros(1)
  time_current = zeros(1)
  Δt = zeros(1)
  Δt = zeros(1)
  fill!(time_start, time_start_in)
  fill!(time_end, time_end_in)
  fill!(time_current, time_start_in)
  fill!(Δt, (time_end_in - time_start_in) / n_steps)
  return TimeStepper(time_start, time_end, time_current, Δt)
  # return TimeStepper(time_start_in, time_end_in, )
end

abstract type AbstractParameters end

struct Parameters{D, N, T, Phys, Props, S, V, H1} <: AbstractParameters
  # actual parameter fields
  # TODO add boundary condition stuff and time stepping stuff
  dirichlet_bcs::D
  neumann_bcs::N
  times::T
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

# TODO 
# 1. need to loop over bcs and vars in dof
#    to make organize different bcs based on fspace type
# 2. group all dbcs, nbcs of similar fspaces into a single struct?
# 3. figure out how to handle function pointers on the GPU
# 4. add different fspace types
# 5. convert vectors of dbcs/nbcs into namedtuples
function Parameters(
  assembler, physics,
  dirichlet_bcs, 
  neumann_bcs, 
  times
)
  h1_dbcs = create_bcs(assembler, H1Field)
  h1_field = create_field(assembler, H1Field)

  # TODO
  properties = nothing
  state_old = nothing
  state_new = nothing

  if dirichlet_bcs !== nothing
    syms = map(x -> Symbol("dirichlet_bc_$x"), 1:length(dirichlet_bcs))
    # dbcs = NamedTuple{tuple(syms...)}(tuple(dbcs...))
    # dbcs = DirichletBCContainer(dbcs, size(assembler.dof.H1_vars[1].fspace.coords, 1))
    dirichlet_bcs = DirichletBCContainer.((assembler.dof,), dirichlet_bcs)
    temp_dofs = mapreduce(x -> x.bookkeeping.dofs, vcat, dirichlet_bcs)
    temp_dofs = unique(sort(temp_dofs))
    dirichlet_bcs = NamedTuple{tuple(syms...)}(tuple(dirichlet_bcs...))
    # TODO eventually do something different here
    # update_dofs!(assembler.dof, dbcs.bookkeeping.dofs)
    update_dofs!(assembler.dof, temp_dofs)
  end

  if neumann_bcs !== nothing
    syms = map(x -> Symbol("neumann_bc_$x"), 1:length(neumann_bcs))
    neumann_bcs = NamedTuple{tuple(syms...)}(tuple(neumann_bcs...))
  end

  # dummy time stepper for a static problem
  if times === nothing
    times = TimeStepper(0., 0., 1)
  end

  p = Parameters(
    dirichlet_bcs,
    neumann_bcs, 
    times,
    physics, 
    properties, 
    state_old, state_new, 
    h1_dbcs, h1_field
  )

  update_dofs!(assembler, p)

  return p
end

function create_parameters(
  assembler, physics; 
  dirichlet_bcs=nothing, 
  neumann_bcs=nothing,
  times=nothing
)
  return Parameters(assembler, physics, dirichlet_bcs, neumann_bcs, times)
end

function update_dofs!(asm::SparseMatrixAssembler, p::Parameters)
  # temp_dofs = mapreduce(x -> x.bookkeeping.dofs, vcat, values(p.dirichlet_bcs))
  update_dofs!(asm, p.dirichlet_bcs)
  Base.resize!(p.h1_dbcs, length(asm.dof.H1_bc_dofs))
  return nothing
end

function update_time!(p::Parameters)
  fill!(p.times.time_current, current_time(p.times) + time_step(p.times))
  return nothing
end
