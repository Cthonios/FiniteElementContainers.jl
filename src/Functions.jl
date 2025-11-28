"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
abstract type AbstractFunction{S, F <: FunctionSpace} end

function Adapt.adapt_structure(to, var::T) where T <: AbstractFunction
  syms = names(var)
  fspace = adapt(to, var.fspace)
  type = eval(T.name.name)
  return type{syms, typeof(fspace)}(fspace)
end

"""
$(TYPEDSIGNATURES)
"""
function Base.length(::AbstractFunction{S, F}) where {S, F}
  if typeof(S) <: Symbol
    return 1
  else
    return length(S)
  end
end

"""
$(TYPEDSIGNATURES)
"""
function Base.names(::AbstractFunction{S, F}) where {S, F}
  if typeof(S) <: Symbol
    return (S,)
  else
    return S
  end
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct ScalarFunction{S, F} <: AbstractFunction{S, F}
  fspace::F
end

"""
$(TYPEDSIGNATURES)
"""
function ScalarFunction(fspace::FunctionSpace, sym)
  return ScalarFunction{sym, typeof(fspace)}(fspace)
end

function Base.show(io::IO, ::ScalarFunction{S, F}) where {S, F}
  println(io, "ScalarFunction:")
  println(io, "  names: $S")
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct VectorFunction{S, F} <: AbstractFunction{S, F}
  fspace::F
end

"""
$(TYPEDSIGNATURES)
"""
function VectorFunction(fspace::FunctionSpace, sym)
  syms = ()
  components = [:_x, :_y, :_z]
  for n in axes(fspace.coords, 1)
    syms = (syms..., String(sym) * String(components[n]))
  end
  syms = Symbol.(syms)
  return VectorFunction{syms, typeof(fspace)}(fspace)
end

function Base.show(io::IO, ::VectorFunction{S, F}) where {S, F}
  println(io, "VectorFunction:")
  println(io, "  names: $S")
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct TensorFunction{S, F} <: AbstractFunction{S, F}
  fspace::F
end

"""
$(TYPEDSIGNATURES)
"""
function TensorFunction(fspace::FunctionSpace, sym; use_spatial_dimension=false)
  # switch for whether to do the sensible thing for constitutive equations
  # e.g. always use 3x3 for displacement gradient regardless of dimension
  if use_spatial_dimension
    ND = size(values(fspace.coords), 1)
  else
    ND = 3
  end

  if ND == 2
    components = [
      :_xx, :_yy, 
      :_xy, :_yx
    ]
  elseif ND == 3
    components = [
      :_xx, :_yy, :_zz, 
      :_yz, :_xz, :_xy, 
      :_zy, :_zx, :_yx
    ]
  else
    @assert false "TensorFunction likely doesn't make sense for ND not 2 or 3"
  end

  syms = ()
  for n in axes(components, 1)
    syms = (syms..., String(sym) * String(components[n]))
  end
  syms = Symbol.(syms)
  return TensorFunction{syms, typeof(fspace)}(fspace)
end

function Base.show(io::IO, ::TensorFunction{S, F}) where {S, F}
  println(io, "TensorFunction:")
  println(io, "  names: $S")
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct SymmetricTensorFunction{S, F} <: AbstractFunction{S, F}
  fspace::F
end

"""
$(TYPEDSIGNATURES)
Uses numbering consistent with exodus output, is this the right thing to do?
Should it be consistent with Tensors.jl
"""
function SymmetricTensorFunction(fspace::FunctionSpace, sym; use_spatial_dimension=false)
  # switch for whether to do the sensible thing for constitutive equations
  # e.g. always use 3x3 for displacement gradient regardless of dimension
  if use_spatial_dimension
    ND = size(values(fspace.coords), 1)
  else
    ND = 3
  end

  # build component symbol extensions
  if ND == 2
    components = [
      :_xx, :_yy, 
      :_xy
    ]
  elseif ND == 3
    components = [
      :_xx, :_yy, :_zz, 
      :_yz, :_xz, :_xy,
    ]
  else
    @assert false "SymmetricTensorFunction likely doesn't make sense for ND not 2 or 3."
  end

  # finally set up component symbols
  syms = ()
  for n in axes(components, 1)
    syms = (syms..., String(sym) * String(components[n]))
  end
  syms = Symbol.(syms)
  return SymmetricTensorFunction{syms, typeof(fspace)}(fspace)
end

function Base.show(io::IO, ::SymmetricTensorFunction{S, F}) where {S, F}
  println(io, "SymmetricTensorFunction:")
  println(io, "  names: $S")
end


# struct StateFunction{S, F, NS, NQ} <: AbstractFunction{S, F}
#   fspace::F
# end

# """
# $(TYPEDSIGNATURES)
# """
# function StateFunction(fspace::FunctionSpace, sym, n_state, n_quad_pts)
#   syms = ()
#   for n in 1:n_state
#     for q in 1:n_quad_pts
#       syms = (syms..., String(sym) * String("_$(n)_$(q)"))
#     end
#   end
#   syms = Symbol.(syms)
#   return StateFunction{syms, typeof(fspace), n_state, n_quad_pts}(fspace)
# end

# function Base.show(io::IO, ::StateFunction{S, F, NS, NQ}) where {S, F, NS, NQ}
#   println(io, "StateFunction:")
#   println(io, "  names: $S")
#   println(io, "  number of state variables: $NS")
#   println(io, "  number of quadrature points: $NQ")
# end
