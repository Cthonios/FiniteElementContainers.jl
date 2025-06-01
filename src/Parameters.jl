abstract type AbstractParameters end

# TODO need to break up bcs to different field types
struct Parameters{
  D, N, T, Phys, Props, S, 
  H1Coords, H1
} <: AbstractParameters
  # parameter/solution fields
  dirichlet_bcs::D
  neumann_bcs::N
  times::T
  physics::Phys
  properties::Props
  state_old::S
  state_new::S
  h1_coords::H1Coords
  h1_field::H1
  # scratch fields
  h1_hvp::H1
end

# TODO 
# 1. need to loop over bcs and vars in dof
#    to make organize different bcs based on fspace type
# 2. group all dbcs, nbcs of similar fspaces into a single struct?
# 3. figure out how to handle function pointers on the GPU - done
# 4. add different fspace types
# 5. convert vectors of dbcs/nbcs into namedtuples - done
function Parameters(
  assembler, physics,
  properties,
  dirichlet_bcs, 
  neumann_bcs, 
  times
)
  h1_coords = assembler.dof.H1_vars[1].fspace.coords
  h1_field = create_field(assembler, H1Field)
  h1_hvp = create_field(assembler, H1Field)

  # TODO
  # properties = nothing

  # for mixed spaces we'll need to do this more carefully
  if isa(physics, AbstractPhysics)
    syms = keys(values(assembler.dof.H1_vars)[1].fspace.elem_conns)
    physics = map(x -> physics, syms)
    physics = NamedTuple{tuple(syms...)}(tuple(physics...))
  else
    @assert isa(physics, NamedTuple)
    # TODO re-arrange physics tuple to match fspaces when appropriate
  end

  if isa(properties, AbstractArray)
    syms = keys(values(assembler.dof.H1_vars)[1].fspace.elem_conns)
    properties = map(x -> properties, syms)
    properties = NamedTuple{tuple(syms...)}(tuple(properties...))
  else
    @assert isa(properties, NamedTuple)
  end

  # state_old = Array{Float64, 3}[]
  # properties = []
  state_old = L2QuadratureField[]
  for (key, val) in pairs(physics)
    # create properties for this block physics
    # TODO specialize to allow for element level properties
    # push!(properties, create_properties(val))

    # create state variables for this block physics
    NS = num_states(val)
    NQ = ReferenceFiniteElements.num_quadrature_points(
      getfield(values(assembler.dof.H1_vars)[1].fspace.ref_fes, key)
    )
    NE = size(
      getfield(values(assembler.dof.H1_vars)[1].fspace.elem_conns, key),
      2
    )

    syms = tuple(map(x -> Symbol("state_variable_$x"), 1:NS)...)
    state_old_temp = zeros(NS, NQ, NE)
    for e in 1:NE
      for q in 1:NQ
        state_old_temp[:, q, e] = create_initial_state(val)
      end
    end
    state_old_temp = L2QuadratureField(state_old_temp, syms)

    push!(state_old, state_old_temp)
  end

  # properties = NamedTuple{keys(physics)}(tuple(properties...))

  state_new = copy(state_old)
  state_old = NamedTuple{keys(physics)}(tuple(state_old...))
  state_new = NamedTuple{keys(physics)}(tuple(state_new...))

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
    h1_coords, h1_field, 
    # scratch fields
    h1_hvp
  )

  update_dofs!(assembler, p)
  Uu = create_unknowns(assembler.dof, H1Field)

  # assemble the stiffness at least once for 
  # making easier to use on GPU
  assemble!(assembler, Uu, p, Val{:stiffness}(), H1Field)
  K = stiffness(assembler)

  return p
end

function Base.show(io::IO, parameters::Parameters)
  println(io, "Parameters:")
  println(io, "Dirichlet Boundary Conditions:")
  for bc in parameters.dirichlet_bcs
    println(io, "$bc")
  end
  println(io, "Neumann Boundary Conditions:")
  for bc in parameters.neumann_bcs
    println(io, "$bc")
  end
  println(io, parameters.times)
  println(io, "Physics:")
  for (physics, props) in zip(parameters.physics, parameters.properties)
    println(io, physics)
    println(io, "Props = $props")
  end
  println("Number of active state variables = $(mapreduce(x -> length(x), sum, values(parameters.state_old)))")
end

function create_parameters(
  assembler, physics, props; 
  dirichlet_bcs=NamedTuple(), 
  neumann_bcs=NamedTuple(),
  times=nothing
)
  return Parameters(assembler, physics, props, dirichlet_bcs, neumann_bcs, times)
end

function update_bcs!(p::Parameters)
  for bc in values(p.dirichlet_bcs)
    _update_bcs!(bc, p.h1_field, KA.get_backend(bc))
  end
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
  return nothing
end

function update_dofs!(asm::SparseMatrixAssembler, p::Parameters)
  update_dofs!(asm, p.dirichlet_bcs)
  return nothing
end

function update_time!(p::Parameters)
  fill!(p.times.time_current, current_time(p.times) + time_step(p.times))
  return nothing
end
