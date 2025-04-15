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
