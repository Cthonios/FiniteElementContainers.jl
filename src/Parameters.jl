struct BlockMismatchError <: AbstractFECError
  msg::String
end
_block_mismatch_error(msg::String) = throw(BlockMismatchError(msg))

function _check_block_keys(given, expected, what)
  extra  = filter(x -> !(x in expected), collect(given))
  absent = filter(x -> !(x in given), collect(expected))
  if !isempty(extra) || !isempty(absent)
    msg = "$what must have exactly one entry per element block.\n" *
          "  element blocks : $(join(expected, ", "))\n" *
          "  $what entries : $(join(given, ", "))"
    isempty(extra)  || (msg *= "\n  not element blocks   : $(join(extra, ", "))")
    isempty(absent) || (msg *= "\n  blocks with no entry : $(join(absent, ", "))")
    _block_mismatch_error(msg)
  end
  return nothing
end

"""
Align a user-supplied `physics`/`properties` argument with the element blocks of
`fspace`, returning a `NamedTuple` keyed by block name and ordered by block
index, so that entry `b` always belongs to block `b`.

Everything downstream -- `_setup_state_variables`, `foreach_block`, the
assembly kernels -- pairs entry `b` with block `b` positionally. A `NamedTuple`
supplied in a different order than the mesh's blocks would therefore hand each
block another block's material without any error, so it is permuted here, and
its keys are required to be exactly the block names.
"""
function _align_blocks(fspace, x::NamedTuple, what)
  names = tuple(Symbol.(block_names(fspace))...)
  _check_block_keys(keys(x), names, what)
  return NamedTuple{names}(map(name -> getfield(x, name), names))
end

# A bare `Tuple` carries no block names, so there is no way to tell whether it
# is in block order or not. Rejecting is the only safe reading.
function _align_blocks(fspace, x::Tuple, what)
  _block_mismatch_error(
    "$what was given as an unnamed Tuple, which cannot be matched to element " *
    "blocks. Supply a NamedTuple keyed by block name " *
    "($(join(block_names(fspace), ", "))), or a single value to share across " *
    "all blocks."
  )
end

# a single physics/properties object shared by every block
function _align_blocks(fspace, x, what)
  names = tuple(Symbol.(block_names(fspace))...)
  return NamedTuple{names}(ntuple(_ -> x, length(names)))
end

# a single properties object shared by every block
# needs to be constant props, can't be element level
# unless we have one block, but let's not specialize that muc
function _setup_properties(fspace, props::Vector)
  return PropertyField(map(_ -> props, block_names(fspace)))
end

# namedtuple case that should become deprecated soon
function _setup_properties(fspace, props::NamedTuple)
  names = tuple(Symbol.(block_names(fspace))...)
  _check_block_keys(keys(props), names, "properties")
  return PropertyField([map(x -> getfield(props, x), names)...])
end

function _setup_properties(fspace, props::Dict{String})
  names = block_names(fspace)
  _check_block_keys(keys(props), names, "properties")
  return PropertyField([map(x -> props[x], names)...])
end

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
  state_old = StateVariableField(state_old)
  state_new = StateVariableField(state_new)
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
  D,       # dimension
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
  PBCFuncs <: AbstractVector,
  RBCFuncs <: AbstractVector,
  Phys,
  Field    <: AbstractField 
} <: AbstractParameters
  ics::InitialConditions{ICFuncs, IV, RV}
  dirichlet_bcs::DirichletBCs{DBCFuncs, IV, RV}
  neumann_bcs::NeumannBCs{NBCFuncs, IT, IV, RM1}
  periodic_bcs::PeriodicBCs{PBCFuncs, IV, RV}
  robin_bcs::RobinBCs{RBCFuncs, IT, IV, RM2, RM3}
  sources::Sources{SRCFuncs, RM4}
  times::TimeStepper{RT}
  physics::Phys
  properties::PropertyField{RT, RV, IV}
  state_old::StateVariableField{RT, RV}
  state_new::StateVariableField{RT, RV}
  coords::H1Field{RT, RV, D}
  field::Field
  field_old::Field
  # scratch fields
  hvp_scratch_field::Field
end
  
function Parameters(
  mesh, assembler,
  physics, properties,
  ics,
  dbcs, nbcs, pbcs, rbcs,
  sources,
  times
)
  dof = assembler.dof
  fspace = function_space(dof)
  coords = coordinates(fspace)

  ics = InitialConditions(mesh, dof, ics)
  dbcs = DirichletBCs(mesh, dof, dbcs)
  nbcs = NeumannBCs(mesh, dof, nbcs)
  pbcs = PeriodicBCs(mesh, dof, pbcs)
  rbcs = RobinBCs(mesh, dof, rbcs)
  sources = Sources(mesh, dof, sources)

  if times === nothing
    times = TimeStepper(0., 0., 1)
  end

  # for mixed spaces we'll need to do this more carefully
  physics = _align_blocks(fspace, physics, "physics")
  properties = _setup_properties(fspace, properties)

  # setup state variables
  state_old, state_new = _setup_state_variables(fspace, physics)

  # scratch
  field = create_field(dof)
  field_old = create_field(dof)
  hvp_scratch_field = create_field(dof)

  # update assembler, where should this really live?
  update_dofs!(assembler, dbcs, pbcs)

  return Parameters(
    ics, dbcs, nbcs, pbcs, rbcs, sources, times,
    physics, properties,
    state_old, state_new,
    coords, field, field_old, hvp_scratch_field
  )
end

function Adapt.adapt_structure(to, p::Parameters)
  return Parameters(
    adapt(to, p.ics),
    adapt(to, p.dirichlet_bcs),
    adapt(to, p.neumann_bcs),
    adapt(to, p.periodic_bcs),
    adapt(to, p.robin_bcs),
    adapt(to, p.sources),
    adapt(to, p.times),
    adapt(to, p.physics),
    adapt(to, p.properties),
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
  println(io, "Periodic Boundary Conditions:")
  println(io, parameters.periodic_bcs)
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
  D,     # dimension
  SFuncT,
  VFuncT,
  IT     <: Integer,
  RT     <: Number,
  IV     <: AbstractVector{IT},
  RV     <: AbstractVector{RT},
  RM     <: AbstractMatrix{<:SVector},
  Phys,
  Field  <: AbstractField 
} <: AbstractParameters
  ics::InitialConditions{Vector{InitialConditionFunction{SFuncT}}, IV, RV}
  dirichlet_bcs::DirichletBCs{Vector{DirichletBCFunction{SFuncT, SFuncT, SFuncT}}, IV, RV}
  neumann_bcs::NeumannBCs{Vector{NeumannBCFunction{VFuncT}}, IT, IV, RM}
  periodic_bcs::PeriodicBCs{Vector{PeriodicBCFunction{SFuncT}}, IV, RV}
  # robin_bcs::RobinBCs{RBCFuncs, IT, IV, RM2, RM3}
  sources::Sources{Vector{SourceFunction{VFuncT}}, RM}
  times::TimeStepper{RT}
  physics::Phys
  properties::PropertyField{RT, RV, IV}
  state_old::StateVariableField{RT, RV}
  state_new::StateVariableField{RT, RV}
  coords::H1Field{RT, RV, D}
  field::Field
  field_old::Field
  # scratch fields
  hvp_scratch_field::Field

  function TypeStableParameters{D, SF, VF}(mesh, assembler, physics, props, ics, dbcs, nbcs, pbcs, srcs, times) where {D, SF, VF}
    dof = assembler.dof
    ND = size(dof, 1)
    fspace = function_space(dof)
    ics = InitialConditions{SF}(mesh, dof, ics)
    dbcs = DirichletBCs{SF}(mesh, dof, dbcs)
    nbcs = NeumannBCs{VF}(mesh, dof, nbcs)
    pbcs = PeriodicBCs{SF}(mesh, dof, pbcs)
    srcs = Sources{VF}(mesh, dof, srcs)

    physics = _align_blocks(fspace, physics, "physics")
    # props = _align_blocks(fspace, props, "properties")
    props = _setup_properties(fspace, props)

    state_old, state_new = _setup_state_variables(fspace, physics)

    coords = mesh.nodal_coords
    field = create_field(assembler)
    field_old = create_field(assembler)
    hvp_scratch_field = create_field(assembler)

    # update assembler, where should this really live?
    update_dofs!(assembler, dbcs, pbcs)

    new{
      D, SF, VF, Int, Float64, Vector{Int}, Vector{Float64}, Matrix{SVector{ND, Float64}},
      typeof(physics), typeof(field)
    }(
      ics, dbcs, nbcs, pbcs, srcs,
      times, 
      physics, props, state_old, state_new, coords, field, field_old, hvp_scratch_field
    )
  end

  function TypeStableParameters{D, SF, VF}(
    mesh, assembler, physics, props, state_old, state_new,
    ics, dbcs, nbcs, pbcs, srcs, times
  ) where {D, SF, VF}
    dof = assembler.dof
    ND = size(dof, 1)
    fspace = function_space(dof)
    ics = InitialConditions{SF}(mesh, dof, ics)
    dbcs = DirichletBCs{SF}(mesh, dof, dbcs)
    nbcs = NeumannBCs{VF}(mesh, dof, nbcs)
    pbcs = PeriodicBCs{SF}(mesh, dof, pbcs)
    srcs = Sources{VF}(mesh, dof, srcs)

    physics = _align_blocks(fspace, physics, "physics")
    # props = _align_blocks(fspace, props, "properties")
    props = _setup_properties(fspace, props)

    coords = mesh.nodal_coords
    field = create_field(assembler)
    field_old = create_field(assembler)
    hvp_scratch_field = create_field(assembler)

    # update assembler, where should this really live?
    update_dofs!(assembler, dbcs, pbcs)

    new{
      D, SF, VF, Int, Float64, Vector{Int}, Vector{Float64}, Matrix{SVector{ND, Float64}},
      typeof(physics), typeof(field)
    }(
      ics, dbcs, nbcs, pbcs, srcs,
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
  periodic_bcs         = PeriodicBC[],
  robin_bcs            = RobinBC[],
  sources              = Source[],
  times                = nothing
)
  return Parameters(
    mesh, assembler, physics, props, ics, 
    dirichlet_bcs, neumann_bcs, periodic_bcs, robin_bcs, sources, times
  )
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
function periodic_dofs(p::AbstractParameters)
  return periodic_dofs(p.periodic_dofs)
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
  update_bc_values!(p.periodic_bcs, X, t)
  # update_bc_values!(p.robin_bcs, assembler, X, t, p.field)
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
  update_bc_values!(p.periodic_bcs, X, t)
  # update_bc_values!(p.robin_bcs, assembler, X, t, p.field)
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
  update_dofs!(asm, p.dirichlet_bcs, asm.periodic_bcs)
  return nothing
end

function _update_for_assembly!(p::AbstractParameters, dof::DofManager, Uu)
  update_field_dirichlet_bcs!(p.field, p.dirichlet_bcs)
  update_field_unknowns!(p.field, dof, Uu)
  # below needs to occur after update_field_unknowns!
  update_field_periodic_bcs!(p.field, p.periodic_bcs)

  # # Robin BC values need to be updated here to be correct
  # update_bc_values!(p.robin_bcs, p.coords, current_time(p), p.field)
  return nothing
end

function _update_for_assembly!(p::AbstractParameters, dof::DofManager, Uu, Vu)
  update_field_dirichlet_bcs!(p.field, p.dirichlet_bcs)
  update_field_unknowns!(p.field, dof, Uu)
  update_field_unknowns!(p.hvp_scratch_field, dof, Vu)
  # below needs to occur after update_field_unknowns!
  update_field_periodic_bcs!(p.field, p.periodic_bcs)

  # # Robin BC values need to be updated here to be correct
  # update_bc_values!(p.robin_bcs, p.coords, current_time(p), p.field)
  return nothing
end

# Full-DOF flavor: caller is responsible for assembling the merged
# vectors U_full = [Uu; U_BC] and v_full = [v_free; v_BC] themselves.
# Unlike the free-DOF flavors above, we do NOT call
# update_field_dirichlet_bcs! — overwriting BC slots would silently
# erase the BC contribution the caller intentionally placed in U_full.
function _update_for_assembly_full!(
  p::AbstractParameters,
  U_full::AbstractVector{<:Number},
  v_full::AbstractVector{<:Number}
)
  @assert length(U_full) == length(p.field.data)
  @assert length(v_full) == length(p.hvp_scratch_field.data)
  copyto!(p.field.data, U_full)
  copyto!(p.hvp_scratch_field.data, v_full)
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function update_time!(p::AbstractParameters)
  p.times.time_current = current_time(p.times) + time_step(p.times)
  return nothing
end
