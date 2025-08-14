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
  blocks = adapt(to, bk.blocks)
  side_nodes = adapt(to, bk.side_nodes)
  return FiniteElementContainers.BCBookKeeping{typeof(side_nodes), typeof(blocks)}(
    blocks,
    adapt(to, bk.dofs),
    adapt(to, bk.elements),
    adapt(to, bk.nodes),
    adapt(to, bk.sides),
    side_nodes
  )
end

function Adapt.adapt_structure(to, bc::FiniteElementContainers.DirichletBCContainer)
  bk = adapt(to, bc.bookkeeping)
  vals = adapt(to, bc.vals)
  return FiniteElementContainers.DirichletBCContainer{
    typeof(bk), eltype(vals), typeof(vals)
  }(
    bk, vals
  )
end

function Adapt.adapt_structure(to, bc::FiniteElementContainers.NeumannBCContainer)
  bk = adapt(to, bc.bookkeeping)
  vals = adapt(to, bc.vals)
  return FiniteElementContainers.NeumannBCContainer{
    typeof(bk), eltype(vals), typeof(vals)
  }(
    bk, vals
  )
end

# DofManagers
function Adapt.adapt_structure(to, dof::DofManager{
  T, IDs, 
  NH1Dofs, NHcurlDofs, NHdivDofs, NL2EDofs, NL2QDofs,
  H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars
}) where {
  T, IDs, 
  NH1Dofs, NHcurlDofs, NHdivDofs, NL2EDofs, NL2QDofs,
  H1Vars, HcurlVars, HdivVars, L2EVars, L2QVars
}
  H1_bc_dofs = adapt(to, dof.H1_bc_dofs)
  H1_unknown_dofs = adapt(to, dof.H1_unknown_dofs)
  Hcurl_bc_dofs = adapt(to, dof.Hcurl_bc_dofs)
  Hcurl_unknown_dofs = adapt(to, dof.Hcurl_unknown_dofs)
  Hdiv_bc_dofs = adapt(to, dof.Hdiv_bc_dofs)
  Hdiv_unknown_dofs = adapt(to, dof.Hdiv_unknown_dofs)
  L2_element_dofs = adapt(to, dof.L2_element_dofs)
  L2_quadrature_dofs = adapt(to, dof.L2_quadrature_dofs)
  H1_vars = adapt(to, dof.H1_vars)
  Hcurl_vars = adapt(to, dof.Hcurl_vars)
  Hdiv_vars = adapt(to, dof.Hdiv_vars)
  L2_element_vars = adapt(to, dof.L2_element_vars)
  L2_quadrature_vars =adapt(to, dof.L2_quadrature_vars)
  return DofManager{
    T, typeof(H1_bc_dofs), 
    NH1Dofs, NHcurlDofs, NHdivDofs, NL2EDofs, NL2QDofs,
    typeof(H1_vars), typeof(Hcurl_vars), typeof(Hdiv_vars), typeof(L2_element_vars), typeof(L2_quadrature_vars)  
  }(
    H1_bc_dofs, H1_unknown_dofs,
    Hcurl_bc_dofs, Hcurl_unknown_dofs,
    Hdiv_bc_dofs, Hdiv_unknown_dofs,
    L2_element_dofs, L2_quadrature_dofs,
    H1_vars, Hcurl_vars, Hdiv_vars, L2_element_vars, L2_quadrature_vars
  )
end

# Fields
function Adapt.adapt_structure(to, field::L2ElementField{T, NF, V, S}) where {T, NF, V, S}
  vals = adapt(to, field.vals)
  return L2ElementField{T, NF, typeof(vals), S}(vals)
end

function Adapt.adapt_structure(to, field::L2QuadratureField{T, NF, NQ, V, S}) where {T, NF, NQ, V, S}
  vals = adapt(to, field.vals)
  return L2QuadratureField{T, NF, NQ, typeof(vals), S}(vals)
end

function Adapt.adapt_structure(to, field::H1Field{T, NF, V, S}) where {T, NF, V, S}
  vals = adapt(to, field.vals)
  return H1Field{T, NF, typeof(vals), S}(vals)
end

# Function spaces
function Adapt.adapt_structure(to, fspace::FunctionSpace)
  coords = adapt(to, fspace.coords)
  elem_conns = adapt(to, fspace.elem_conns)
  elem_id_maps = adapt(to, fspace.elem_id_maps)
  ref_fes = adapt(to, fspace.ref_fes)
  sideset_elems = adapt(to, fspace.sideset_elems)
  sideset_nodes = adapt(to, fspace.sideset_nodes)
  sideset_sides = adapt(to, fspace.sideset_sides)
  sideset_side_nodes = adapt(to, fspace.sideset_side_nodes)
  return FunctionSpace(
    coords, 
    elem_conns, elem_id_maps, 
    ref_fes,
    sideset_elems, 
    sideset_nodes, 
    sideset_sides, sideset_side_nodes
  )
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
