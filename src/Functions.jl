abstract type AbstractFunction{S, F <: FunctionSpace} end
function Base.length(::AbstractFunction{S, F}) where {S, F}
  if typeof(S) <: Symbol
    return 1
  else
    return length(S)
  end
end
function Base.names(::AbstractFunction{S, F}) where {S, F}
  if typeof(S) <: Symbol
    return (S,)
  else
    return S
  end
end

struct ScalarFunction{S, F} <: AbstractFunction{S, F}
  fspace::F
end

function ScalarFunction(fspace::FunctionSpace, sym)
  return ScalarFunction{sym, typeof(fspace)}(fspace)
end

function Base.show(io::IO, ::ScalarFunction{S, F}) where {S, F}
  println(io, "ScalarFunction:")
  println(io, "  name: $S")
end

struct VectorFunction{S, F} <: AbstractFunction{S, F}
  fspace::F
end

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
  println(io, "  name: $S")
end

struct TensorFunction{S, F} <: AbstractFunction{S, F}
  fspace::F
end

function TensorFunction(fspace::FunctionSpace, sym)
  syms = ()
  if size(fspace.coords, 1) == 2
    components = [
      :_xx, :_yy, 
      :_xy, :_yx
    ]
  elseif size(fspace.coords, 1) == 3
    components = [
      :_xx, :_yy, :_zz, 
      :_yz, :_xz, :_xy, 
      :_zy, :_zx, :_yx
    ]
  end
  for n in axes(components, 1)
    syms = (syms..., String(sym) * String(components[n]))
  end
  syms = Symbol.(syms)
  return TensorFunction{syms, typeof(fspace)}(fspace)
end

function Base.show(io::IO, ::TensorFunction{S, F}) where {S, F}
  println(io, "TensorFunction:")
  println(io, "  name: $S")
end

struct SymmetricTensorFunction{S, F} <: AbstractFunction{S, F}
  fspace::F
end

function SymmetricTensorFunction(fspace::FunctionSpace, sym)
  syms = ()
  if size(fspace.coords, 1) == 2
    components = [
      :_xx, :_yy, 
      :_xy
    ]
  elseif size(fspace.coords, 1) == 3
    components = [
      :_xx, :_yy, :_zz, 
      :_yz, :_xz, :_xy,
    ]
  end
  for n in axes(components, 1)
    syms = (syms..., String(sym) * String(components[n]))
  end
  syms = Symbol.(syms)
  return TensorFunction{syms, typeof(fspace)}(fspace)
end

function Base.show(io::IO, ::SymmetricTensorFunction{S, F}) where {S, F}
  println(io, "TensorFunction:")
  println(io, "  name: $S")
end
