"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
User facing API to define a ```DirichletBC````.
"""
struct DirichletBC{F} <: AbstractBC{F}
  func::F
  sset_name::Symbol
  var_name::Symbol
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
function DirichletBC(var_name::Symbol, sset_name::Symbol, func::Function)
  return DirichletBC{typeof(func)}(func, sset_name, var_name)
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
function DirichletBC(var_name::String, sset_name::String, func::Function)
  return DirichletBC(Symbol(var_name), Symbol(sset_name), func)
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
Internal implementation of dirichlet BCs
"""
struct DirichletBCContainer{B, V} <: AbstractBCContainer{B, V}
  bookkeeping::B
  # func::F
  vals::V
end

# TODO modify as follows
# 1. remove bookkeeping from bcs
# 2. have bcs only take in a sset name, var name, and func
# 3. create one giant bookkeeper here

function DirichletBCContainer(dof::DofManager, dbc::DirichletBC)
  var_name = dbc.var_name
  sset_name = dbc.sset_name

  bk = BCBookKeeping(dof, var_name, sset_name)

  # now sort and unique this stuff
  dof_perm = _unique_sort_perm(bk.dofs)
  el_perm = _unique_sort_perm(bk.elements)

  # do permutations
  dofs_new = bk.dofs[dof_perm]
  elements_new = bk.elements[el_perm]
  nodes_new = bk.nodes[dof_perm]
  sides_new = bk.sides[el_perm]
  resize!(bk.dofs, length(dofs_new))
  resize!(bk.elements, length(elements_new))
  resize!(bk.nodes, length(nodes_new))
  resize!(bk.sides, length(sides_new))
  copyto!(bk.dofs, dofs_new)
  copyto!(bk.elements, elements_new)
  copyto!(bk.nodes, nodes_new)
  copyto!(bk.sides, sides_new)

  vals = zeros(length(bk.nodes))

  # return DirichletBCContainer(bk, dbc.func, vals)
  return DirichletBCContainer(bk, vals)
end

KA.get_backend(x::DirichletBCContainer) = KA.get_backend(x.bookkeeping)
