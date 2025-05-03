struct NeumannBC{F} <: AbstractBC{F}
  func::F
  sset_name::Symbol
  var_name::Symbol
end

# TODO need to hack the var_name thing
function NeumannBC(var_name::Symbol, sset_name::Symbol, func::Function)
  return NeumannBC{typeof(func)}(func, sset_name, var_name)
end
