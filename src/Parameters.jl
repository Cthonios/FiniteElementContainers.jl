function _setup_state_variables(fspace, physics)
  state_old = Array{Float64, 3}[]
  state_new = Array{Float64, 3}[]
  for (b, val) in enumerate(values(physics))
    # create state variables for this block physics
    NS = num_states(val)
    NQ, NE = block_quadrature_size(fspace, b)

    state_old_temp = zeros(NS, NQ, NE)
    state_new_temp = zeros(NS, NQ, NE)
    for e in 1:NE
      for q in 1:NQ
        state_old_temp[:, q, e] = create_initial_state(val)
        state_new_temp[:, q, e] = create_initial_state(val)
      end
    end
    push!(state_old, state_old_temp)
    push!(state_new, state_new_temp)
  end
  state_old = L2Field(state_old)
  state_new = L2Field(state_new)
  return state_old, state_new
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
abstract type AbstractParameters end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
struct Parameters{
  IT       <: Integer,
  RT       <: Number,
  IV       <: AbstractVector{IT},
  RV       <: AbstractVector{RT},
  RM1      <: AbstractMatrix,
  RM2      <: AbstractMatrix,
  RM3      <: AbstractMatrix,
  RM4      <: AbstractMatrix,
  ICFuncs  <: AbstractVector,
  DBCFuncs <: AbstractVector,
  SRCFuncs <: AbstractVector,
  NBCFuncs <: AbstractVector,
  RBCFuncs <: AbstractVector,
  Phys,
  Props,
  Coords   <: AbstractField,
  Field    <: AbstractField 
} <: AbstractParameters
  ics::InitialConditions{ICFuncs, IV, RV}
  dirichlet_bcs::DirichletBCs{DBCFuncs, IV, RV}
  neumann_bcs::NeumannBCs{NBCFuncs, IT, IV, RM1}
  robin_bcs::RobinBCs{RBCFuncs, IT, IV, RM2, RM3}
  sources::Sources{SRCFuncs, RM4}
  times::TimeStepper{RT}
  physics::Phys
  properties::Props
  state_old::L2Field{RT, RV}
  state_new::L2Field{RT, RV}
  coords::Coords
  field::Field
  field_old::Field
  # scratch fields
  hvp_scratch_field::Field
end
  
function Parameters(
  mesh, assembler,
  physics, properties,
  ics,
  dbcs, nbcs, rbcs,
  sources,
  times
)
  dof = assembler.dof
  fspace = function_space(dof)
  coords = coordinates(fspace)

  ics = InitialConditions(mesh, dof, ics)
  dbcs = DirichletBCs(mesh, dof, dbcs)
  nbcs = NeumannBCs(mesh, dof, nbcs)
  rbcs = RobinBCs(mesh, dof, rbcs)
  sources = Sources(mesh, dof, sources)

  if times === nothing
    times = TimeStepper(0., 0., 1)
  end

  # for mixed spaces we'll need to do this more carefully
  if isa(physics, AbstractPhysics)
    syms = map(x -> Symbol("region_$x"), 1:length(fspace.ref_fes))
    physics = map(x -> physics, syms)
    physics = NamedTuple{tuple(syms...)}(tuple(physics...))
  else
    @assert isa(physics, NamedTuple)
    # TODO re-arrange physics tuple to match fspaces when appropriate
  end

  if isa(properties, AbstractArray)
    syms = map(x -> Symbol("region_$x"), 1:length(fspace.ref_fes))
    properties = map(x -> properties, syms)
    properties = NamedTuple{tuple(syms...)}(tuple(properties...))
  else
    @assert isa(properties, NamedTuple)
  end

  # setup state variables
  state_old, state_new = _setup_state_variables(fspace, physics)

  # scratch
  field = create_field(dof)
  field_old = create_field(dof)
  hvp_scratch_field = create_field(dof)

  # update assembler, where should this really live?
  update_dofs!(assembler, dbcs)

  return Parameters(
    ics, dbcs, nbcs, rbcs, sources, times,
    physics, properties,
    state_old, state_new,
    coords, field, field_old, hvp_scratch_field
  )
end

function Adapt.adapt_structure(to, p::Parameters)

  # need to handle props specially
  props = []
  for p in values(p.properties)
    if isa(p, SArray)
      push!(props, p)
    else
      push!(props, adapt(to, p))
    end
  end

  props = NamedTuple{keys(p.properties)}(props)

  return Parameters(
    adapt(to, p.ics),
    adapt(to, p.dirichlet_bcs),
    adapt(to, p.neumann_bcs),
    adapt(to, p.robin_bcs),
    adapt(to, p.sources),
    adapt(to, p.times),
    adapt(to, p.physics),
    props,
    adapt(to, p.state_old),
    adapt(to, p.state_new),
    adapt(to, p.coords),
    adapt(to, p.field),
    adapt(to, p.field_old),
    # scratch fields
    adapt(to, p.hvp_scratch_field)
  )
end

function Base.show(io::IO, parameters::Parameters)
  println(io, "Parameters:")
  println(io, "Initial Conditions:")
  println(io, parameters.ics)
  println(io, "Dirichlet Boundary Conditions:")
  println(io, parameters.dirichlet_bcs)
  println(io, "Neumann Boundary Conditions:")
  println(io, parameters.neumann_bcs)
  println(io, "Robin Boundary Conditions:")
  println(io, parameters.robin_bcs)
  println(io, "Sources:")
  println(io, parameters.sources)
  println(io, parameters.times)
  println(io, "Physics:")
  for (physics, props) in zip(parameters.physics, parameters.properties)
    println(io, physics)
    println(io, "Props = $props")
  end
  println("Number of active state variables = $(length(parameters.state_old.data))")
end

function KA.get_backend(p::Parameters)
  return KA.get_backend(p.field)
end

struct TypeStableParameters{
  # Funcs  <: AbstractVector,
  FuncT,
  IT     <: Integer,
  RT     <: Number,
  IV     <: AbstractVector{IT},
  RV     <: AbstractVector{RT},
  RM     <: AbstractMatrix{<:SVector},
  Phys,
  Props,
  Coords <: AbstractField,
  Field  <: AbstractField 
} <: AbstractParameters
  ics::InitialConditions{Vector{InitialConditionFunction{FuncT}}, IV, RV}
  dirichlet_bcs::DirichletBCs{Vector{DirichletBCFunction{FuncT, FuncT, FuncT}}, IV, RV}
  neumann_bcs::NeumannBCs{Vector{NeumannBCFunction{FuncT}}, IT, IV, RM}
  # robin_bcs::RobinBCs{RBCFuncs, IT, IV, RM2, RM3}
  sources::Sources{Vector{SourceFunction{FuncT}}, RM}
  times::TimeStepper{RT}
  physics::Phys
  properties::Props
  state_old::L2Field{RT, RV}
  state_new::L2Field{RT, RV}
  coords::Coords
  field::Field
  field_old::Field
  # scratch fields
  hvp_scratch_field::Field

  function TypeStableParameters{F}(mesh, assembler, physics, props, ics, dbcs, nbcs, srcs, times) where F
    dof = assembler.dof
    ND = size(dof, 1)
    fspace = function_space(dof)
    ics = InitialConditions{F}(mesh, dof, ics)
    dbcs = DirichletBCs{F}(mesh, dof, dbcs)
    nbcs = NeumannBCs{F}(mesh, dof, nbcs)
    srcs = Sources{F}(mesh, dof, srcs)

    state_old, state_new = _setup_state_variables(fspace, physics)

    coords = mesh.nodal_coords
    field = create_field(assembler)
    field_old = create_field(assembler)
    hvp_scratch_field = create_field(assembler)

    # update assembler, where should this really live?
    update_dofs!(assembler, dbcs)

    new{
      F, Int, Float64, Vector{Int}, Vector{Float64}, Matrix{SVector{ND, Float64}},
      typeof(physics), typeof(props), typeof(mesh.nodal_coords), typeof(field)
    }(
      ics, dbcs, nbcs, srcs,
      times, 
      physics, props, state_old, state_new, coords, field, field_old, hvp_scratch_field
    )
  end
end

function create_parameters(
  mesh, assembler, physics, props;
  ics                  = InitialCondition[],
  dirichlet_bcs        = DirichletBC[],
  neumann_bcs          = NeumannBC[],
  robin_bcs            = RobinBC[],
  sources              = Source[],
  times                = nothing
)
  return Parameters(mesh, assembler, physics, props, ics, dirichlet_bcs, neumann_bcs, robin_bcs, sources, times)
end

"""
$(TYPEDSIGNATURES)
"""
function coordinates(p::AbstractParameters)
  return p.coords
end

"""
$(TYPEDSIGNATURES)
"""
function current_time(p::AbstractParameters)
  return current_time(p.times)
end

"""
$(TYPEDSIGNATURES)
"""
function dirichlet_dofs(p::AbstractParameters)
  return dirichlet_dofs(p.dirichlet_bcs)
end

"""
$(TYPEDSIGNATURES)
"""
function initialize!(p::AbstractParameters)
  update_ic_values!(p.ics, coordinates(p))
  update_field_ics!(p.field, p.ics)
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function time_step(p::AbstractParameters)
  return time_step(p.times)
end

"""
$(TYPEDSIGNATURES)
This method is used to update the stored bc values.
This should be called at the beginning of any load step

This method only handles updating bc values
for Dirichlet and Neumann BCs

Robin BC updates are handled in robin assembly method
"""
function update_bc_values!(p::AbstractParameters, assembler)
  X = coordinates(p)
  t = current_time(p)
  update_bc_values!(p.dirichlet_bcs, X, t)
  update_bc_values!(p.neumann_bcs, assembler, X, t)
  update_source_values!(p.sources, assembler, X, t)

  # TODO how to handle Robin BCs?
  # currently assembly methods handle updating the field
  # in parameters with the current unknown dofs
  # we need field here to reflect that for the robin bcs
  # to be correct...

  # order of operations goes
  return nothing
end

function update_bc_values!(p::TypeStableParameters, assembler)
  X = coordinates(p)
  t = current_time(p)
  update_bc_values!(p.dirichlet_bcs, X, t)
  update_bc_values!(p.neumann_bcs, assembler, X, t)
  update_source_values!(p.sources, assembler, X, t)

  # TODO how to handle Robin BCs?
  # currently assembly methods handle updating the field
  # in parameters with the current unknown dofs
  # we need field here to reflect that for the robin bcs
  # to be correct...

  # order of operations goes
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function update_dofs!(asm::AbstractAssembler, p::Parameters)
  update_dofs!(asm, p.dirichlet_bcs)
  return nothing
end

function _update_for_assembly!(p::AbstractParameters, dof::DofManager, Uu)
  update_field_dirichlet_bcs!(p.field, p.dirichlet_bcs)
  update_field_unknowns!(p.field, dof, Uu)

  # # Robin BC values need to be updated here to be correct
  # update_bc_values!(p.robin_bcs, p.coords, current_time(p), p.field)
  return nothing
end

function _update_for_assembly!(p::AbstractParameters, dof::DofManager, Uu, Vu)
  update_field_dirichlet_bcs!(p.field, p.dirichlet_bcs)
  update_field_unknowns!(p.field, dof, Uu)
  update_field_unknowns!(p.hvp_scratch_field, dof, Vu)

  # # Robin BC values need to be updated here to be correct
  # update_bc_values!(p.robin_bcs, p.coords, current_time(p), p.field)
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function update_time!(p::AbstractParameters)
  p.times.time_current = current_time(p.times) + time_step(p.times)
  return nothing
end
