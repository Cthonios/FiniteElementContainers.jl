abstract type AbstractFunctionSpaceType end

struct H1 <: AbstractFunctionSpaceType
end

struct Hcurl <: AbstractFunctionSpaceType
end

struct Hdiv <: AbstractFunctionSpaceType
end

struct L2Element <: AbstractFunctionSpaceType
end

struct L2Quadrature <: AbstractFunctionSpaceType
end

struct FunctionSpace{Coords, ElemConns, ElemIdMaps, RefFEs, FSpaceType}
  coords::Coords
  elem_conns::ElemConns
  elem_id_maps::ElemIdMaps
  ref_fes::RefFEs
  fspace_type::FSpaceType
end

function FunctionSpace(mesh::AbstractMesh, ::Type{H1}, interp_type, q_degree)
  # block_names = Symbol.(valkeys(mesh.element_conns))
  block_names = mesh.element_block_names
  ref_fes = ReferenceFE[]
  for elem_name in mesh.element_types
    elem_type = elem_type_map[elem_name]
    ref_fe = ReferenceFE(elem_type{interp_type, q_degree}())
    push!(ref_fes, ref_fe)
  end
  ref_fes = NamedTuple{tuple(block_names...)}(tuple(ref_fes...))

  if interp_type == Lagrange
    coords = mesh.nodal_coords
  else
    throw(ErrorException("Unssuported interp_type $interp_type"))
  end

  return FunctionSpace(coords, mesh.element_conns, mesh.element_id_maps, ref_fes, H1())
end

function FunctionSpace(mesh::AbstractMesh, ::Type{L2Element}, interp_type)
  block_names = mesh.element_block_names
  ref_fes = ReferenceFE[]
  for elem_name in mesh.element_types
    elem_type = elem_type_map[elem_name]
    ref_fe = ReferenceFE(elem_type{interp_type, 1}())
    push!(ref_fes, ref_fe)
  end
  ref_fes = NamedTuple{tuple(block_names...)}(tuple(ref_fes...))

  # TODO need to map nodal coordinates from mesh to element centroids


end

function FunctionSpace(mesh::AbstractMesh, space_type, interp_type; q_degree=2)
  return FunctionSpace(mesh, space_type, interp_type, q_degree)
end

function Base.show(io::IO, fspace::FunctionSpace)
  println(io, "FunctionSpace:")
  println(io, "  Type: $(fspace.fspace_type)")
  for (key, ref_fe) in enumerate(fspace.ref_fes)
    println(io, "    Block: $key")
    # println(io, "      Number of elements: $(fspace)")
    # println(io, "  $ref_fe")
  end
end

function connectivity(fspace, block)
  return Base.getproperty(fspace.elem_conns, block)
end

function _connectivity(fspace, e::Int, block)
  temp = @views Base.getproperty(fspace.elem_conns, block)[:, e]
  # NN = num_vertices(fspace.ref_fes[block])
  # return SVector{NN, eltype(temp)}(temp)
  return temp
end

connectivity(fspace, e, block::Symbol) = _connectivity(fspace, e, Val(block))

function connectivity(fspace, n::Int, e::Int, block)
  return @views Base.getproperty(fspace.elem_conns, block)[n, e]
end 

coordinates(fspace::FunctionSpace) = fspace.coords

function coordinates(fspace::FunctionSpace, e, block)
  conn = _connectivity(fspace, e, block)
  return @views fspace.coords[:, conn]
end

# function scalar_function(fspace::FunctionSpace)
#   vals = zeros(size(fspace.coords, 2))
#   return NodalField{Float64, 1, Vector{Float64}}(vals)
# end

# function vector_function(fspace::FunctionSpace)
#   vals = zeros(size(fspace.coords, 1) * size(fspace.coords, 2))
#   return NodalField{Float64, size(fspace.coords, 1), Vector{Float64}}(vals)
# end

# used to construct functions
# const ScalarFunction = Symbol

abstract type AbstractFunction{S, F <: FunctionSpace} end
function Base.length(::AbstractFunction{S, F}) where {S, F}
  if typeof(S) <: Symbol
    return 1
  else
    return length(S)
  end
end
Base.names(::AbstractFunction{S, F}) where {S, F} = S

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
