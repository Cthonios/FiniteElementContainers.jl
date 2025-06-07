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
struct DirichletBCContainer{B, T, V} <: AbstractBCContainer{B, T, V}
  bookkeeping::B
  vals::V
end

"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
function DirichletBCContainer(dof::DofManager, dbc::DirichletBC)
  bk = BCBookKeeping(dof, dbc.var_name, dbc.sset_name)
  vals = zeros(length(bk.nodes))
  return DirichletBCContainer{typeof(bk), eltype(vals), typeof(vals)}(
    bk, vals
  )
end
