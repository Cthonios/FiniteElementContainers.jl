"""
"""
abstract type AbstractField{T, N, NFields, Names, Vals} <: AbstractArray{T, N} end

"""
"""
field_names(field::AbstractField) = field.names

"""
"""
num_fields(::AbstractField{T, N, NFields, Names, Vals}) where {T, N, NFields, Names, Vals} = NFields

Base.IndexStyle(::Type{<:AbstractField})            = IndexLinear()
Base.size(field::AbstractField)                     = size(field.vals)
Base.getindex(field::AbstractField, index::Int)     = getindex(field.vals, index)
Base.setindex!(field::AbstractField, v, index::Int) = setindex!(field.vals, v, index)
Base.axes(field::AbstractField)                     = Base.axes(field.vals)

##################################################################################

"""
"""
struct NodalField{T, N, NFields, NNodes, Names, Vals} <: AbstractField{T, N, NFields, Names, Vals}
  names::Names
  vals::Vals
end

"""
"""
function NodalField{1, NNodes}(names, vals::V) where {NNodes, V <: Vector}
  @assert NNodes == length(vals)
  return NodalField{eltype(vals), ndims(vals), 1, NNodes, typeof(names), typeof(vals)}(names, vals)
end

"""
"""
function NodalField{NFields, NNodes}(names, vals::M) where {NFields, NNodes, M <: Matrix}
  @assert NFields == size(vals, 1)
  @assert NNodes  == size(vals, 2)
  return NodalField{eltype(vals), ndims(vals), NFields, NNodes, typeof(names), typeof(vals)}(names, vals)
end

"""
"""
function NodalField{NFields, NNodes}(::Type{A}, ::Type{T}, names, ::UndefInitializer) where {NFields, NNodes, A <: AbstractArray, T}
  if A <: Vector
    @assert NFields == 1
    vals = A{T}(undef, NNodes)
  elseif A <: Matrix
    vals = A{T}(undef, NFields, NNodes)
  else
    @assert false "Unsupported type in NodalField undef constructor. TODO add an exception"
  end

  return NodalField{eltype(vals), ndims(vals), NFields, NNodes, typeof(names), typeof(vals)}(names, vals)
end

"""
"""
num_nodes(::NodalField{T, N, NFields, NNodes, Names, Vals}) where {T, N, NFields, NNodes, Names, Vals} = NNodes

Base.show(io::IO, field::N) where N <: NodalField = print(io,
  "\nNodal fields named \"$(field_names(field))\" with $(num_fields(field)) fields and $(num_nodes(field)) nodes.\n",
  "Values = "
)

###################################################################################

"""
"""
struct ElementField{T, N, NFields, NElements, Names, Vals} <: AbstractField{T, N, NFields, Names, Vals}
  names::Names
  vals::Vals
end

"""
"""
function ElementField{1, NElements}(names, vals::V) where {NElements, V <: Vector}
  @assert NElements == length(vals)
  return ElementField{eltype(vals), ndims(vals), 1, NElements, typeof(names), typeof(vals)}(names, vals)
end

"""
"""
function ElementField{NFields, NElements}(names, vals::M) where {NFields, NElements, M <: Matrix}
  @assert NFields   == size(vals, 1)
  @assert NElements == size(vals, 2)
  return ElementField{eltype(vals), ndims(vals), NFields, NElements, typeof(names), typeof(vals)}(names, vals)
end

"""
"""
function ElementField{NFields, NElements}(
  ::Type{A}, ::Type{T}, 
  names, ::UndefInitializer
) where {NFields, NElements, A <: AbstractArray, T}
  # TODO fill this out more
  if A <: Vector
    @assert NFields == 1
    vals = A{T}(undef, NElements)
  elseif A <: Matrix
    vals = A{T}(undef, NFields, NElements)
  elseif A <: StructArray
    if T <: Union{<:MArray, <:SArray}
      @assert NFields == length(T)
      vals = A{T}(undef, NElements)
    else
      @assert false "Unsupported type in ElementField undef constructor."
    end
  else
    @show A
    @show T
    @assert false "Unsupported type in ElementField undef constructor."
  end
  # vals = A{T}(undef, NFields, NElements)
  return ElementField{eltype(vals), ndims(vals), NFields, NElements, typeof(names), typeof(vals)}(names, vals)
end

"""
"""
function ElementField{NFields, NElements}(names, vals::S) where {NFields, NElements, S <: StructArray}
  @assert ndims(vals) == 1
  @assert NFields     == length(vals[1])
  @assert NElements   == length(vals)
  return ElementField{eltype(vals), ndims(vals), NFields, NElements, typeof(names), typeof(vals)}(names, vals)
end


"""
"""
num_elements(::ElementField{T, N, NFields, NElements, Names, Vals}) where {T, N, NFields, NElements, Names, Vals} = NElements

Base.show(io::IO, field::N) where N <: ElementField = print(io, 
  "\nElement fields named \"$(field_names(field))\" with $(num_fields(field)) fields and $(num_elements(field)) elements.\n",
  "Values = "
)

###################################################################################

"""
"""
struct QuadratureField{T, N, NFields, NQPoints, NElements, Names, Vals} <: AbstractField{T, N, NFields, Names, Vals}
  names::Names
  vals::Vals
end

"""
"""
function QuadratureField{NF, NQ, NE}(names, vals::M) where {NF, NQ, NE, M <: Matrix}
  @assert NF == 1
  @assert NQ == size(vals, 1)
  @assert NE == size(vals, 2)
  return QuadratureField{eltype(vals), ndims(vals), NF, NQ, NE, typeof(names), typeof(vals)}(names, vals)
end

"""
"""
function QuadratureField{NF, NQ, NE}(names, vals::A) where {NF, NQ, NE, A <: Array{<:Number, 3}}
  @assert NF == size(vals, 1)
  @assert NQ == size(vals, 2)
  @assert NE == size(vals, 3)
  return QuadratureField{eltype(vals), ndims(vals), NF, NQ, NE, typeof(names), typeof(vals)}(names, vals)
end

"""
"""
function QuadratureField{NF, NQ, NE}(names, vals::S) where {NF, NQ, NE, S <: StructArray}
  @assert NF == length(vals[1])
  @assert NQ == size(vals, 1)
  @assert NE == size(vals, 2)
  return QuadratureField{eltype(vals), ndims(vals), NF, NQ, NE, typeof(names), typeof(vals)}(names, vals)
end

"""
"""
function QuadratureField{NF, NQ, NE}(
  ::Type{A}, ::Type{T}, 
  names, ::UndefInitializer
) where {NF, NQ, NE, A <: Union{<:Matrix, <:StructArray}, T}
  
  if T <: Union{MArray, SArray}
    @assert NF == length(T)
  end

  vals = A{T}(undef, NQ, NE)
  return QuadratureField{eltype(vals), ndims(vals), NF, NQ, NE, typeof(names), typeof(vals)}(names, vals)
end

"""
"""
num_elements(::QuadratureField{T, N, NFields, NQPoints, NElements, Names, Vals}) where {T, N, NFields, NQPoints, NElements, Names, Vals} = NElements

"""
"""
num_q_points(::QuadratureField{T, N, NFields, NQPoints, NElements, Names, Vals}) where {T, N, NFields, NQPoints, NElements, Names, Vals} = NQPoints

Base.show(io::IO, field::N) where N <: QuadratureField = print(io, 
  "\nQuadrature fields named \"$(field_names(field))\" with $(num_fields(field)) fields and $(num_elements(field)) elements.\n",
  "Values = "
)
