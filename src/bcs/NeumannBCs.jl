struct NeumannBC{S, B, F, V} <: AbstractBC{S, B, F, V}
  bookkeeping::B
  func::F
  vals::V
end

# TODO need to hack the var_name thing
function NeumannBC(dof::DofManager, var_name::Symbol, sset_name::Symbol, func::Function)
  bookkeeping = BCBookKeeping(dof, var_name, sset_name)
  vals = zeros(Float64, length(bookkeeping.elements))
  sym = Symbol(var_name, :_, sset_name) # TODO maybe add func name?
  return DirichletBC{sym, typeof(bookkeeping), typeof(func), typeof(vals)}(bookkeeping, func, vals)
end
