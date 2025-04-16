struct NeumannBC{S, B, F} <: AbstractBC{S, B, F}
  bookkeeping::B
  func::F
end

# TODO need to hack the var_name thing
function NeumannBC(dof::DofManager, var_name::Symbol, sset_name::Symbol, func::Function)
  bookkeeping = BCBookKeeping(dof, var_name, sset_name)
  sym = Symbol(var_name, :_, sset_name) # TODO maybe add func name?
  return DirichletBC{sym, typeof(bookkeeping), typeof(func)}(bookkeeping, func)
end
