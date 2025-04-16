struct DirichletBC{S, B, F} <: AbstractBC{S, B, F}
  bookkeeping::B
  func::F
end

function DirichletBC(dof::DofManager, var_name::Symbol, sset_name::Symbol, func::Function)
  bookkeeping = BCBookKeeping(dof, var_name, sset_name; build_dofs_array=true)
  sym = Symbol(var_name, :_, sset_name)
  func = BCFunction(func)
  return DirichletBC{sym, typeof(bookkeeping), typeof(func)}(bookkeeping, func)
end

struct DirichletBCContainer{B, F, I, V} <: AbstractBCContainer{B, F, I, V}
  bookkeeping::B
  funcs::F
  func_ids::I
  vals::V
end

# TODO modify as follows
# 1. remove bookkeeping from bcs
# 2. have bcs only take in a sset name, var name, and func
# 3. create one giant bookkeeper here
function DirichletBCContainer(dbcs, num_dim)

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

num_dimensions(bc::DirichletBCContainer) = num_dimensions(bc.bookkeeping)
KA.get_backend(x::DirichletBCContainer) = KA.get_backend(x.bookkeeping)
