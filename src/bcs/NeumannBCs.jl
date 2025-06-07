"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
User facing API to define a ```NeumannBC````.
"""
struct NeumannBC{F} <: AbstractBC{F}
  func::F
  sset_name::Symbol
  var_name::Symbol
end

# TODO need to hack the var_name thing
"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
function NeumannBC(var_name::Symbol, sset_name::Symbol, func::Function)
  return NeumannBC{typeof(func)}(func, sset_name, var_name)
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
function NeumannBC(var_name::String, sset_name::Symbol, func::Function)
  return NeumannBC{typeof(func)}(Symbol(var_name), Symbol(sset_name), func)
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
Internal implementation of dirichlet BCs
"""
struct NeumannBCContainer{B, T, V} <: AbstractBCContainer{B, T, V}
  bookkeeping::B
  vals::V
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
function NeumannBCContainer(dof::DofManager, nbc::NeumannBC)
  bk = BCBookKeeping(dof, nbc.var_name, nbc.sset_name)
  ND = num_dimensions(dof.H1_vars[1].fspace, keys(dof.H1_vars[1].fspace.elem_id_maps)[1])
  vals = zeros(SVector{ND, Float64}, length(bk.sides))
  return NeumannBCContainer{typeof(bk), eltype(vals), typeof(vals)}(
    bk, vals
  )
end