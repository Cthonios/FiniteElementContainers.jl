# struct DirichletBC{S, B, F} <: AbstractBC{S, B, F}
#   bookkeeping::B
#   func::F
# end

# function DirichletBC(dof::DofManager, var_name::Symbol, sset_name::Symbol, func::Function)
#   bookkeeping = BCBookKeeping(dof, var_name, sset_name; build_dofs_array=true)
#   sym = Symbol(var_name, :_, sset_name)
#   func = BCFunction(func)
#   return DirichletBC{sym, typeof(bookkeeping), typeof(func)}(bookkeeping, func)
# end

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

# struct DirichletBCContainer_old{B, F, I, V} <: AbstractBCContainer{B, F, I, V}
#   bookkeeping::B
#   funcs::F
#   func_ids::I
#   vals::V
# end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
Internal implementation of dirichlet BCs
"""
struct DirichletBCContainer{B, F, V} <: AbstractBCContainer{B, F, V}
  bookkeeping::B
  func::F
  vals::V
end

# TODO modify as follows
# 1. remove bookkeeping from bcs
# 2. have bcs only take in a sset name, var name, and func
# 3. create one giant bookkeeper here
function DirichletBCContainer_old_old(dbcs, num_dim)

  # quick hack fix for now
  

  blocks = mapreduce(x -> repeat(x.bookkeeping.blocks, length(x.bookkeeping.elements)), vcat, dbcs)
  dofs = mapreduce(x -> x.bookkeeping.dofs, vcat, dbcs)
  elements = mapreduce(x -> x.bookkeeping.elements, vcat, dbcs)
  nodes = mapreduce(x -> x.bookkeeping.nodes, vcat, dbcs)
  sides = mapreduce(x -> x.bookkeeping.sides, vcat, dbcs)
  func_ids = mapreduce(
    (x, y) -> fill(x, length(y.bookkeeping.dofs)), 
    vcat, 
    1:length(dbcs), dbcs
  )
  vals = zeros(length(dofs))

  dof_perm = _unique_sort_perm(dofs)
  el_perm = _unique_sort_perm(elements)
  blocks = blocks[el_perm]
  dofs = dofs[dof_perm]
  elements = elements[el_perm]
  nodes = nodes[dof_perm]
  sides = sides[el_perm]
  func_ids = func_ids[dof_perm]
  vals = vals[dof_perm]

  sym = Symbol("dirichlet_bcs")
  bk = BCBookKeeping{num_dim, sym, Int, typeof(dofs)}(blocks, dofs, elements, nodes, sides)
  func_syms = map(x -> Symbol("function_$x"), 1:length(dbcs))
  funcs = map(x -> x.func, dbcs)
  funcs = NamedTuple{tuple(func_syms...)}(tuple(funcs...))

  return DirichletBCContainer(bk, funcs, func_ids, vals)
end

function DirichletBCContainer_old(dof, dbcs)
  var_names = map(x -> x.var_name, dbcs)
  sset_names = map(x -> x.sset_name, dbcs)
  bk = BCBookKeeping(dof, var_names, sset_names)

  # now sort and unique this stuff
  # TODO clean this up or move func ids into bookkeeping
  dof_perm = _unique_sort_perm(bk.dofs)
  el_perm = _unique_sort_perm(bk.elements)

  # TODO fix up blocks
  # copyto!(bk.dofs, bk.dofs[dof_perm])
  # copyto!(bk.elements, bk.elements[el_perm])
  # copyto!(bk.nodes, bk.nodes[dof_perm])
  # copyto!(bk.sides, bk.sides[el_perm])
  # bk.dofs .= bk.dofs[dof_perm]
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
  # bk.dofs .= dofs_new

  # setup func ids
  fspace = dof.H1_vars[1].fspace
  n_entries = map(x -> length(getproperty(fspace.sideset_nodes, x.sset_name)), dbcs)
  func_ids = mapreduce((x, y) -> fill(x, y), vcat, 1:length(n_entries), n_entries)
  func_ids = func_ids[dof_perm]

  func_syms = map(x -> Symbol("function_$x"), 1:length(dbcs))
  funcs = map(x -> x.func, dbcs)
  funcs = NamedTuple{tuple(func_syms...)}(tuple(funcs...))

  vals = zeros(length(bk.nodes))

  return DirichletBCContainer(bk, funcs, func_ids, vals)
end

function DirichletBCContainer(dof, dbc)
  # var_names = map(x -> x.var_name, dbcs)
  # sset_names = map(x -> x.sset_name, dbcs)
  var_name = dbc.var_name
  sset_name = dbc.sset_name
  # bk = BCBookKeeping(dof, var_names, sset_names)
  bk = BCBookKeeping(dof, var_name, sset_name)

  # now sort and unique this stuff
  # TODO clean this up or move func ids into bookkeeping
  dof_perm = _unique_sort_perm(bk.dofs)
  el_perm = _unique_sort_perm(bk.elements)

  # TODO fix up blocks
  # copyto!(bk.dofs, bk.dofs[dof_perm])
  # copyto!(bk.elements, bk.elements[el_perm])
  # copyto!(bk.nodes, bk.nodes[dof_perm])
  # copyto!(bk.sides, bk.sides[el_perm])
  # bk.dofs .= bk.dofs[dof_perm]
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
  # bk.dofs .= dofs_new

  # setup func ids
  # fspace = dof.H1_vars[1].fspace
  # n_entries = map(x -> length(getproperty(fspace.sideset_nodes, x.sset_name)), dbcs)
  # func_ids = mapreduce((x, y) -> fill(x, y), vcat, 1:length(n_entries), n_entries)
  # func_ids = func_ids[dof_perm]

  # func_syms = map(x -> Symbol("function_$x"), 1:length(dbcs))
  # funcs = map(x -> x.func, dbcs)
  # funcs = NamedTuple{tuple(func_syms...)}(tuple(funcs...))

  func = dbc.func

  vals = zeros(length(bk.nodes))

  # return DirichletBCContainer(bk, funcs, func_ids, vals)
  return DirichletBCContainer(bk, func, vals)
end

num_dimensions(bc::DirichletBCContainer) = num_dimensions(bc.bookkeeping)
KA.get_backend(x::DirichletBCContainer) = KA.get_backend(x.bookkeeping)
