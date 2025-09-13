module FiniteElementContainersAdaptExt

using Adapt
using FiniteElementContainers
using StaticArrays

FiniteElementContainersAdaptExt.cpu(x) = adapt(Array, x)

# Assemblers
function Adapt.adapt_structure(to, asm::SparseMatrixAssembler)
  return SparseMatrixAssembler(
    adapt(to, asm.dof),
    adapt(to, asm.pattern),
    adapt(to, asm.constraint_storage),
    adapt(to, asm.damping_storage),
    adapt(to, asm.hessian_storage),
    adapt(to, asm.mass_storage),
    adapt(to, asm.residual_storage),
    adapt(to, asm.residual_unknowns),
    adapt(to, asm.scalar_quadarature_storage),
    adapt(to, asm.stiffness_storage),
    adapt(to, asm.stiffness_action_storage),
    adapt(to, asm.stiffness_action_unknowns)
  )
end

function Adapt.adapt_structure(to, asm::FiniteElementContainers.SparsityPattern)
  Is = adapt(to, asm.Is)
  Js = adapt(to, asm.Js)
  unknown_dofs = adapt(to, asm.unknown_dofs)
  block_sizes = adapt(to, asm.block_sizes)
  block_offsets = adapt(to, asm.block_offsets)
  #
  klasttouch = adapt(to, asm.klasttouch)
  csrrowptr = adapt(to, asm.csrrowptr)
  csrcolval = adapt(to, asm.csrcolval)
  csrnzval = adapt(to, asm.csrnzval)
  #
  csccolptr = adapt(to, asm.csccolptr)
  cscrowval = adapt(to, asm.cscrowval)
  cscnzval = adapt(to, asm.cscnzval)
  return FiniteElementContainers.SparsityPattern(
    Is, Js,
    unknown_dofs, block_sizes, block_offsets,
    klasttouch, csrrowptr, csrcolval, csrnzval,
    csccolptr, cscrowval, cscnzval
  )
end

# Boundary Conditions
function Adapt.adapt_structure(to, bk::FiniteElementContainers.BCBookKeeping{V}) where V
  return FiniteElementContainers.BCBookKeeping(
    adapt(to, bk.blocks),
    adapt(to, bk.dofs),
    adapt(to, bk.elements),
    adapt(to, bk.nodes),
    adapt(to, bk.sides),
    adapt(to, bk.side_nodes)
  )
end

function Adapt.adapt_structure(to, bc::FiniteElementContainers.DirichletBCContainer)
  bk = adapt(to, bc.bookkeeping)
  vals = adapt(to, bc.vals)
  vals_dot = adapt(to, bc.vals_dot)
  vals_dot_dot = adapt(to, bc.vals_dot_dot)
  return FiniteElementContainers.DirichletBCContainer(bk, vals, vals_dot, vals_dot_dot)
end

function Adapt.adapt_structure(to, bc::FiniteElementContainers.NeumannBCContainer)
  bk = adapt(to, bc.bookkeeping)
  el_conns = adapt(to, bc.element_conns)
  surf_conns = adapt(to, bc.surface_conns)
  ref_fe = adapt(to, bc.ref_fe)
  vals = adapt(to, bc.vals)
  return FiniteElementContainers.NeumannBCContainer(bk, el_conns, surf_conns, ref_fe, vals)
end

# DofManagers
function Adapt.adapt_structure(to, dof::DofManager{C, IT, IDs, Var}) where {C, IT, IDs, Var}
  dirichlet_dofs = adapt(to, dof.dirichlet_dofs)
  unknowns = adapt(to, dof.unknown_dofs)
  var = adapt(to, dof.var)
  return DofManager{
    C, IT, typeof(dirichlet_dofs), typeof(var) 
  }(dirichlet_dofs, unknowns, var)
end

# Fields
function Adapt.adapt_structure(to, field::L2ElementField{T, D, NF}) where {T, D, NF}
  data = adapt(to, field.data)
  return L2ElementField{T, typeof(data), NF}(data)
end

function Adapt.adapt_structure(to, field::L2QuadratureField{T, D, NF, NQ}) where {T, D, NF, NQ}
  data = adapt(to, field.data)
  return L2QuadratureField{T, typeof(data), NF, NQ}(data)
end

function Adapt.adapt_structure(to, field::H1Field{T, D, NF}) where {T, D, NF}
  data = adapt(to, field.data)
  return H1Field{T, typeof(data), NF}(data)
end

# Function spaces
function Adapt.adapt_structure(to, fspace::FunctionSpace)
  coords = adapt(to, fspace.coords)
  elem_conns = adapt(to, fspace.elem_conns)
  ref_fes = adapt(to, fspace.ref_fes)
  return FunctionSpace(coords, elem_conns, ref_fes)
end

function Adapt.adapt_structure(to, var::T) where T <: FiniteElementContainers.AbstractFunction
  syms = names(var)
  fspace = adapt(to, var.fspace)
  type = eval(T.name.name)
  return type{syms, typeof(fspace)}(fspace)
end

# parameters
function Adapt.adapt_structure(to, p::FiniteElementContainers.Parameters)

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

  return FiniteElementContainers.Parameters(
    adapt(to, p.dirichlet_bcs),
    adapt(to, p.dirichlet_bc_funcs),
    adapt(to, p.neumann_bcs),
    adapt(to, p.neumann_bc_funcs),
    adapt(to, p.times),
    adapt(to, p.physics),
    props,
    adapt(to, p.state_old),
    adapt(to, p.state_new),
    adapt(to, p.h1_coords),
    adapt(to, p.h1_field),
    adapt(to, p.h1_field_old),
    # scratch fields
    adapt(to, p.h1_hvp_scratch_field)
  )
end

function Adapt.adapt_structure(to, p::FiniteElementContainers.TimeStepper)
  return FiniteElementContainers.TimeStepper(
    adapt(to, p.time_start),
    adapt(to, p.time_end),
    adapt(to, p.time_current),
    adapt(to, p.Δt)
  )
end

end # module
