"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
abstract type AbstractFunction{F <: FunctionSpace, NF} end
function_space(func::AbstractFunction) = func.fspace
num_fields(::AbstractFunction{F, NF}) where {F, NF} = NF

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
struct ScalarFunction{F} <: AbstractFunction{F, 1}
  fspace::F
  names::NTuple{1, String}
end

"""
$(TYPEDSIGNATURES)
"""
function ScalarFunction(fspace::FunctionSpace, sym::String)
  return ScalarFunction(fspace, (sym,))
end

function Adapt.adapt_structure(to, f::ScalarFunction)
  return ScalarFunction(adapt(to, f.fspace), f.names)
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct VectorFunction{F, NF} <: AbstractFunction{F, NF}
  fspace::F
  names::NTuple{NF, String}
end

function Adapt.adapt_structure(to, f::VectorFunction)
  return VectorFunction(adapt(to, f.fspace), f.names)
end

"""
$(TYPEDSIGNATURES)
"""
function VectorFunction(fspace::FunctionSpace, sym::String)
  components = ("_x", "_y", "_z")
  syms = map(x -> "$sym$x", components[1:size(fspace.coords, 1)])
  return VectorFunction(fspace, syms)
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct TensorFunction{F, NF} <: AbstractFunction{F, NF}
  fspace::F
  names::NTuple{NF, String}
end

"""
$(TYPEDSIGNATURES)
"""
function TensorFunction(fspace::FunctionSpace, sym::String; use_spatial_dimension=false)
  # switch for whether to do the sensible thing for constitutive equations
  # e.g. always use 3x3 for displacement gradient regardless of dimension
  if use_spatial_dimension
    ND = size(values(fspace.coords), 1)
  else
    ND = 3
  end

  if ND == 2
    components = (
      "_xx", "_yy",
      "_xy", "_yx"
    )
  elseif ND == 3
    components = (
      "_xx", "_yy", "_zz", 
      "_yz", "_xz", "_xy", 
      "_zy", "_zx", "_yx"
    )
  else
    @assert false "TensorFunction likely doesn't make sense for ND not 2 or 3"
  end

  syms = map(x -> "$sym$x", components)
  return TensorFunction(fspace, syms)
end

function Adapt.adapt_structure(to, f::TensorFunction)
  return TensorFunction(adapt(to, f.fspace), f.names)
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct SymmetricTensorFunction{F, NF} <: AbstractFunction{F, NF}
  fspace::F
  names::NTuple{NF, String}
end

"""
$(TYPEDSIGNATURES)
Uses numbering consistent with exodus output, is this the right thing to do?
Should it be consistent with Tensors.jl
"""
function SymmetricTensorFunction(fspace::FunctionSpace, sym::String; use_spatial_dimension=false)
  # switch for whether to do the sensible thing for constitutive equations
  # e.g. always use 3x3 for displacement gradient regardless of dimension
  if use_spatial_dimension
    ND = size(values(fspace.coords), 1)
  else
    ND = 3
  end

  # build component symbol extensions
  if ND == 2
    components = (
      "_xx", "_yy", 
      "_xy"
    )
  elseif ND == 3
    components = (
      "_xx", "_yy", "_zz", 
      "_yz", "_xz", "_xy",
    )
  else
    @assert false "SymmetricTensorFunction likely doesn't make sense for ND not 2 or 3."
  end

  # finally set up component symbols
  syms = map(x -> "$sym$x", components)
  return SymmetricTensorFunction(fspace, syms)
end

function Adapt.adapt_structure(to, f::SymmetricTensorFunction)
  return SymmetricTensorFunction(adapt(to, f.fspace), f.names)
end

struct GeneralFunction{F, NF} <: AbstractFunction{F, NF}
  fspace::F
  names::NTuple{NF, String}

  # function GeneralFunction(fspace::FunctionSpace, names::NTuple{NF, String}) where NF
  #   new{typeof(fspace), NF}(fspace, names)
  # end
end

function GeneralFunction(args...)
  fspace = args[1].fspace
  syms = args[1].names
  for arg in args
    @assert typeof(arg.fspace) == typeof(fspace)
  end
  # syms = mapreduce(x -> x.names, vcat, args)
  if length(args) > 1
    for n in 2:length(args)
      syms = (syms..., args[n].names...)
    end
  end
  # return GeneralFunction(fspace, syms)
  return GeneralFunction{typeof(fspace), length(syms)}(fspace, syms)
end

function Adapt.adapt_structure(to, f::GeneralFunction)
  return GeneralFunction(adapt(to, f.fspace), f.names)
end
