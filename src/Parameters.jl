"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
abstract type AbstractParameters end

# TODO need to break up bcs to different field types
"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
struct Parameters{
  RT <: Number, # Real type
  RV <: AbstractArray{RT, 1}, # Real vector type
  IV <: AbstractArray{<:Integer, 1},
  ICFuncs <: NamedTuple,
  DBCFuncs <: NamedTuple,
  NBCs <: NeumannBCs,
  Phys <: NamedTuple, 
  Props <: NamedTuple,  
  NDims,
  NH1Fields
} <: AbstractParameters
  # parameter/solution fields
  ics::InitialConditions{IV, RV, ICFuncs}
  dirichlet_bcs::DirichletBCs{IV, RV, DBCFuncs}
  neumann_bcs::NBCs
  times::TimeStepper{RV}
  physics::Phys
  properties::Props
  state_old::L2Field{RT, RV}
  state_new::L2Field{RT, RV}
  h1_coords::H1Field{RT, RV, NDims}
  h1_field::H1Field{RT, RV, NH1Fields}
  h1_field_old::H1Field{RT, RV, NH1Fields}
  # scratch fields
  h1_hvp_scratch_field::H1Field{RT, RV, NH1Fields}
end

# TODO 
# 1. need to loop over bcs and vars in dof
#    to make organize different bcs based on fspace type
# 2. group all dbcs, nbcs of similar fspaces into a single struct?
# 3. figure out how to handle function pointers on the GPU - done
# 4. add different fspace types
# 5. convert vectors of dbcs/nbcs into namedtuples - done
"""
$(TYPEDSIGNATURES)
"""
function Parameters(
  mesh, assembler, physics,
  properties,
  ics,
  dirichlet_bcs, 
  neumann_bcs, 
  times
)
  # h1_coords = assembler.dof.H1_vars[1].fspace.coords
  h1_coords = function_space(assembler.dof).coords
  h1_field = create_field(assembler)
  h1_field_old = create_field(assembler)
  h1_hvp = create_field(assembler)

  # for mixed spaces we'll need to do this more carefully
  if isa(physics, AbstractPhysics)
    syms = keys(function_space(assembler.dof).ref_fes)
    physics = map(x -> physics, syms)
    physics = NamedTuple{tuple(syms...)}(tuple(physics...))
  else
    @assert isa(physics, NamedTuple)
    # TODO re-arrange physics tuple to match fspaces when appropriate
  end

  if isa(properties, AbstractArray)
    syms = keys(function_space(assembler.dof).ref_fes)
    properties = map(x -> properties, syms)
    properties = NamedTuple{tuple(syms...)}(tuple(properties...))
  else
    @assert isa(properties, NamedTuple)
  end

  state_old = Array{Float64, 3}[]
  for (b, (key, val)) in enumerate(pairs(physics))
    # create state variables for this block physics
    NS = num_states(val)
    NQ = ReferenceFiniteElements.num_quadrature_points(
      getfield(function_space(assembler.dof).ref_fes, key)
    )
    NE = num_elements(function_space(assembler.dof), b)

    state_old_temp = zeros(NS, NQ, NE)
    for e in 1:NE
      for q in 1:NQ
        state_old_temp[:, q, e] = create_initial_state(val)
      end
    end
    push!(state_old, state_old_temp)
  end

  state_new = deepcopy(state_old)
  # state_old = NamedTuple{keys(physics)}(tuple(state_old...))
  # state_new = NamedTuple{keys(physics)}(tuple(state_new...))
  
  state_old = L2Field(state_old)
  state_new = L2Field(state_new)

  ics = InitialConditions(
    mesh, assembler.dof, ics
  )
  dirichlet_bcs = DirichletBCs(
    mesh, assembler.dof, dirichlet_bcs
  )
  neumann_bcs = NeumannBCs(
    mesh, assembler.dof, neumann_bcs
  )

  # dummy time stepper for a static problem
  if times === nothing
    times = TimeStepper(0., 0., 1)
  end

  p = Parameters(
    ics,
    dirichlet_bcs,
    neumann_bcs,
    times,
    physics, 
    properties, 
    state_old, state_new, 
    h1_coords, h1_field, h1_field_old,
    # scratch fields
    h1_hvp
  )

  update_dofs!(assembler, p)
  # Uu = create_unknowns(assembler.dof)

  # # assemble the stiffness at least once for 
  # # making easier to use on GPU
  # # TODO should we also assemble mass if necessary?
  # assemble_stiffness!(assembler, stiffness, Uu, p)
  # K = stiffness(assembler)

  return p
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
    adapt(to, p.times),
    adapt(to, p.physics),
    props, # TODO this will need an adapt when you get to element level props
    adapt(to, p.state_old),
    adapt(to, p.state_new),
    adapt(to, p.h1_coords),
    adapt(to, p.h1_field),
    adapt(to, p.h1_field_old),
    # scratch fields
    adapt(to, p.h1_hvp_scratch_field)
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
  println(io, parameters.times)
  println(io, "Physics:")
  for (physics, props) in zip(parameters.physics, parameters.properties)
    println(io, physics)
    println(io, "Props = $props")
  end
  println("Number of active state variables = $(length(parameters.state_old.data))")
end

function create_parameters(
  mesh, assembler, physics, props; 
  ics=InitialCondition[],
  dirichlet_bcs=DirichletBC[], 
  neumann_bcs=NeumannBC[],
  times=nothing
)
  return Parameters(mesh, assembler, physics, props, ics, dirichlet_bcs, neumann_bcs, times)
end

"""
$(TYPEDSIGNATURES)
"""
function initialize!(p::Parameters)
  update_ic_values!(p.ics, p.h1_coords)
  update_field_ics!(p.h1_field, p.ics)
  return nothing
end

"""
$(TYPEDSIGNATURES)
This method is used to update the stored bc values.
This should be called at the beginning of any load step

TODO need to incorporate other bcs besides H1 spaces
TODO need to incorporate neumann bc updates
"""
function update_bc_values!(p::Parameters)
  X = p.h1_coords
  t = current_time(p.times)
  update_bc_values!(p.dirichlet_bcs, X, t)
  update_bc_values!(p.neumann_bcs, X, t)
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function update_dofs!(asm::AbstractAssembler, p::Parameters)
  update_dofs!(asm, p.dirichlet_bcs)
  return nothing
end

function _update_for_assembly!(p::Parameters, dof::DofManager, Uu)
  update_field_dirichlet_bcs!(p.h1_field, p.dirichlet_bcs)
  update_field_unknowns!(p.h1_field, dof, Uu)
  return nothing
end

function _update_for_assembly!(p::Parameters, dof::DofManager, Uu, Vu)
  update_field_dirichlet_bcs!(p.h1_field, p.dirichlet_bcs)
  update_field_unknowns!(p.h1_field, dof, Uu)
  update_field_unknowns!(p.h1_hvp_scratch_field, dof, Vu)
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function update_time!(p::Parameters)
  fill!(p.times.time_current, current_time(p.times) + time_step(p.times))
  return nothing
end
