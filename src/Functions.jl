"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
abstract type AbstractFunction{F <: FunctionSpace} end

function Adapt.adapt_structure(to, var::T) where T <: AbstractFunction
  fspace = adapt(to, var.fspace)
  type = eval(T.name.name)
  return type{typeof(fspace)}(fspace, var.names)
end

"""
$(TYPEDSIGNATURES)
"""
Base.length(func::AbstractFunction) = length(func.names)

"""
$(TYPEDSIGNATURES)
"""
Base.names(func::AbstractFunction) = func.names

function Base.show(io::IO, func::AbstractFunction)
  println(io, "$(typeof(func).name.name):")
  println(io, "  names: $(names(func))")
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct ScalarFunction{F} <: AbstractFunction{F}
  fspace::F
  names::Vector{Symbol}
end

"""
$(TYPEDSIGNATURES)
"""
function ScalarFunction(fspace::FunctionSpace, sym::Symbol)
  return ScalarFunction(fspace, Symbol[sym])
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct VectorFunction{F} <: AbstractFunction{F}
  fspace::F
  names::Vector{Symbol}
end

"""
$(TYPEDSIGNATURES)
"""
function VectorFunction(fspace::FunctionSpace, sym::Symbol)
  components = [:_x, :_y, :_z]
  syms = map(x -> Symbol("$sym$x"), components[1:size(fspace.coords, 1)])
  return VectorFunction(fspace, syms)
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct TensorFunction{F} <: AbstractFunction{F}
  fspace::F
  names::Vector{Symbol}
end

"""
$(TYPEDSIGNATURES)
"""
function TensorFunction(fspace::FunctionSpace, sym::Symbol; use_spatial_dimension=false)
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

  syms = map(x -> Symbol("$sym$x"), components)
  return TensorFunction(fspace, syms)
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct SymmetricTensorFunction{F} <: AbstractFunction{F}
  fspace::F
  names::Vector{Symbol}
end

"""
$(TYPEDSIGNATURES)
Uses numbering consistent with exodus output, is this the right thing to do?
Should it be consistent with Tensors.jl
"""
function SymmetricTensorFunction(fspace::FunctionSpace, sym::Symbol; use_spatial_dimension=false)
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
  syms = map(x -> Symbol("$sym$x"), components)
  return SymmetricTensorFunction(fspace, syms)
end

struct GeneralFunction{F} <: AbstractFunction{F}
  fspace::F
  names::Vector{Symbol}
end

function GeneralFunction(args...)
  fspace = args[1].fspace
  for arg in args
    @assert typeof(arg.fspace) == typeof(fspace)
  end
  syms = mapreduce(x -> x.names, vcat, args)
  return GeneralFunction(fspace, syms)
end
