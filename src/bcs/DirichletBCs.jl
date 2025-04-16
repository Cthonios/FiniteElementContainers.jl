struct DirichletBC{S, B, F, V} <: AbstractBC{S, B, F, V}
  bookkeeping::B
  func::F
  vals::V
end

function DirichletBC(dof::DofManager, var_name::Symbol, sset_name::Symbol, func::Function)
  bookkeeping = BCBookKeeping(dof, var_name, sset_name; build_dofs_array=true)
  vals = zeros(Float64, length(bookkeeping.nodes))
  sym = Symbol(var_name, :_, sset_name)
  func = BCFunction(func)
  return DirichletBC{sym, typeof(bookkeeping), typeof(func), typeof(vals)}(bookkeeping, func, vals)
end

function _unique_sort_perm(array::AbstractArray{T, 1}) where T <: Number
  ids = unique(i -> array[i], 1:length(array))
  unique_array = array[ids]
  perm = sortperm!(ids, unique_array)
  perm
end

struct DirichletBCContainer{B, F, I, V}
  bookkeeping::B
  funcs::F
  func_ids::I
  vals::V
end

function DirichletBCContainer(dbcs, num_dim)
  blocks = mapreduce(x -> repeat(x.bookkeeping.blocks, length(x.bookkeeping.elements)), vcat, dbcs)
  dofs = mapreduce(x -> x.bookkeeping.dofs, vcat, dbcs)
  elements = mapreduce(x -> x.bookkeeping.elements, vcat, dbcs)
  nodes = mapreduce(x -> x.bookkeeping.nodes, vcat, dbcs)
  sides = mapreduce(x -> x.bookkeeping.sides, vcat, dbcs)
  # func_ids = map((x, y) -> repeat(length(y.bookkeeping.dofs), x), 1:length(dbcs), dbcs) |> vcat
  # func_ids = map((x, y) -> )
  func_ids = mapreduce(
    (x, y) -> fill(x, length(y.bookkeeping.dofs)), 
    vcat, 
    1:length(dbcs), dbcs
  )
  vals = zeros(length(dofs))

  # perm = sortperm(dofs)
  dof_perm = _unique_sort_perm(dofs)
  el_perm = _unique_sort_perm(elements)
  blocks = blocks[el_perm]
  dofs = dofs[dof_perm]
  # @show maximum(dofs)
  # @show length(dofs)
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

KA.get_backend(x::DirichletBCContainer) = KA.get_backend(x.bookkeeping)
