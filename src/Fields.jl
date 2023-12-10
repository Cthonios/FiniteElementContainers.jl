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

#######################################################
# Exceptions
#######################################################

struct FieldTypeException{F <: AbstractField, T1, T2} <: Exception
  field_type::Type{F}
  provided_type::T1
  expected_type::T2
end

Base.show(io::IO, e::FieldTypeException) = 
println(io, 
  "\nField type error type while trying to construct $(e.field_type).", "\n",
  "Provided type = ", e.provided_type, "\n",
  "Expected type = ", e.expected_type, "\n"
)

field_type_error(field_type::Type{<:AbstractField}, provided_type::Type, expected_type::Type) = 
throw(FieldTypeException(field_type, provided_type, expected_type))

struct FieldNumberException{F <: AbstractField, T} <: Exception
  field_type::Type{F}
  provided_type::T
  provided_number_of_fields::Int
  expected_number_of_fields::Int
end

Base.show(io::IO, e::FieldNumberException) =
println(io,
  "\nInvalid number of fields while trying to construct $(e.field_type).\n",
  "Provided type                    = $(e.provided_type)\n",
  "Number of field in provided type = $(e.provided_number_of_fields)\n",
  "Expected number of fields        = $(e.expected_number_of_fields)"
)

function field_number_error(field_type::Type{<:AbstractField}, provided_type::Type, expected_length::Int)
  if provided_type <: Number
    provided_length = ndims(provided_type)
  else
    provided_length = length(provided_type)
  end

  throw(FieldNumberException(field_type, provided_type, provided_length, expected_length))
end

#######################################################
# Nodal field stuff
#######################################################
"""
"""
struct NodalField{T, N, NFields, NNodes, Names, Vals} <: AbstractField{T, N, NFields, Names, Vals}
  names::Names
  vals::Vals
end

Base.size(::NodalField{T, N, NFields, NNodes, Names, Vals}) where {
  T, N, NFields, NNodes, Names, Vals <: AbstractArray{T, 1}
} = (NFields, NNodes)

function Base.getindex(field::NodalField{T, N, NFields, NNodes, Names, Vals}, d::Int, n::Int) where {
  T, N, NFields, NNodes, Names, Vals <: AbstractArray{T, 1}
} 
  @assert d > 0        "Field index out of range in vectorized getindex"
  @assert n > 0        "Field index out of range in vectorized getindex"
  @assert d <= NFields "Field index out of range in vectorized getindex"
  @assert n <= NNodes  "Field index out of range in vectorized getindex"
  getindex(field.vals, (n - 1) * NFields + d)
end

function Base.setindex!(field::NodalField{T, N, NFields, NNodes, Names, Vals}, v, d::Int, n::Int) where {
  T, N, NFields, NNodes, Names, Vals <: AbstractArray{T, 1}
} 

  @assert d > 0        "Field index out of range in vectorized getindex"
  @assert n > 0        "Field index out of range in vectorized getindex"
  @assert d <= NFields "Field index out of range in vectorized getindex"
  @assert n <= NNodes  "Field index out of range in vectorized getindex"
  setindex!(field.vals, v, (n - 1) * NFields + d)
end

Base.axes(::NodalField{T, N, NFields, NNodes, Names, Vals}, ::Val{1}) where {
  T, N, NFields, NNodes, Names, Vals <: AbstractArray{T, 1}
} = Base.OneTo(NFields)

Base.axes(::NodalField{T, N, NFields, NNodes, Names, Vals}, ::Val{2}) where {
  T, N, NFields, NNodes, Names, Vals <: AbstractArray{T, 1}
} = Base.OneTo(NNodes)

function Base.axes(u::NodalField{T, N, NFields, NNodes, Names, Vals}, n::Int) where {
  T, N, NFields, NNodes, Names, Vals <: AbstractArray{T, 1}
} 
  @assert n > 0  "Dimension index out of range in vectorized axes"
  @assert n <= 2 "Dimension index out of range in vectorized axes"
  axes(u, Val(n))
end

"""
"""
function NodalField{NFields, NNodes}(vals::Vector{<:Number}, names) where {NFields, NNodes}
  if NFields == 1
    @assert length(vals) == NNodes
  else
    @assert length(vals) == NNodes * NFields
  end
  return NodalField{eltype(vals), ndims(vals), NFields, NNodes, typeof(names), typeof(vals)}(names, vals)
end

"""
"""
function NodalField{NFields, NNodes}(vals::Matrix{<:Number}, names) where {NFields, NNodes}
  return NodalField{eltype(vals), ndims(vals), NFields, NNodes, typeof(names), typeof(vals)}(names, vals)
end

"""
"""
function NodalField{NFields, NNodes, A, T}(::UndefInitializer, names) where {NFields, NNodes, A <: AbstractArray, T}
  if A <: Vector
    # NOTE in this case we are allowing for a long multi dof vector
    # e.g. convert a 3 x N matrix to a 3N vector
    vals = A{T}(undef, NFields * NNodes)
  elseif A <: Matrix
    vals = A{T}(undef, NFields, NNodes)
  else
    field_type_error(NodalField, A, AbstractArray)
  end
  return NodalField{eltype(vals), ndims(vals), NFields, NNodes, typeof(names), typeof(vals)}(names, vals)
end

"""
"""
function Base.zeros(::Type{NodalField{NFields, NNodes, A, T}}, name) where {NFields, NNodes, A <: AbstractArray, T}
  field = NodalField{NFields, NNodes, A, T}(undef, name)
  field .= zero(eltype(T))
  return field
end

"""
"""
num_nodes(::NodalField{T, N, NFields, NNodes, Names, Vals}) where {T, N, NFields, NNodes, Names, Vals} = NNodes

Base.show(io::IO, field::N) where N <: NodalField = print(io,
  "\nNodal fields named \"$(field_names(field))\" with $(num_fields(field)) fields and $(num_nodes(field)) nodes.\n",
  "Values = "
)

#######################################################
# Element field stuff - TODO this is the least tested likely, it mainly serves
# the function of projecting nodal solution vectors to element containers
#######################################################
"""
"""
struct ElementField{T, N, NFields, NElements, Names, Vals} <: AbstractField{T, N, NFields, Names, Vals}
  names::Names
  vals::Vals
end

"""
"""
function ElementField{NFields, NElements, A, T}(::UndefInitializer, names) where {NFields, NElements, A <: AbstractArray, T}
  
  # scalar variable case
  if T <: Number
    if A <: Vector
      vals = A{T}(undef, NElements)
    elseif A <: Matrix
      vals = A{T}(undef, NFields, NElements)
    else
      if NFields > 1
        type = Vector
      else
        type = Matrix
      end
      field_type_error(ElementField, A, type)
    end
  # non-scalar case
  elseif T <: AbstractArray
    # edge case
    if A <: Matrix || A <: Array{3}
      field_type_error(ElementField, A, AbstractArray{1})
    end

    if NFields != length(T)
      field_number_error(ElementField, T, NFields)
    end
    vals = A{T}(undef, NElements)
  # weird stuff
  else
    field_type_error(ElementField, T, Union{<:Number, <:AbstractArray})
  end
  
  return ElementField{eltype(vals), ndims(vals), NFields, NElements, typeof(names), typeof(vals)}(names, vals)
end

function zeros!(field::ElementField)
  for n in axes(field, 1)
    field[n] = zero(eltype(field))
  end
end

"""
"""
function ElementField{NFields, NElements}(vals::Matrix{<:Number}, names) where {NFields, NElements}
  @assert size(vals, 1) == NFields
  @assert size(vals, 2) == NElements
  return ElementField{eltype(vals), ndims(vals), NFields, NElements, typeof(names), typeof(vals)}(names, vals)
end

"""
"""
function ElementField{NFields, NElements}(vals::S, names) where S <: StructArray where {NFields, NElements}
  @assert size(vals) |> length == 1
  @assert length(vals) == NElements
  @assert length(eltype(vals)) == NFields
  return ElementField{eltype(vals), ndims(vals), NFields, NElements, typeof(names), typeof(vals)}(names, vals)
end

# too many allocations
# """
# """
# function ElementField{NFields, NElements}(vals::A, names) where {NFields, NElements, A <: Base.ReinterpretArray}
#   return ElementField{eltype(vals), ndims(vals), NFields, NElements, typeof(names), typeof(vals)}(names, vals)
# end

"""
"""
function Base.zeros(::Type{ElementField{NFields, NElements, A, T}}, name) where {NFields, NElements, A <: AbstractArray, T}
  field = ElementField{NFields, NElements, A, T}(undef, name)
  zeros!(field)
  return field
end

"""
"""
num_elements(::ElementField{T, N, NFields, NElements, Names, Vals}) where {T, N, NFields, NElements, Names, Vals} = NElements

Base.show(io::IO, field::N) where N <: ElementField = print(io, 
  "\nElement fields named \"$(field_names(field))\" with $(num_fields(field)) fields and $(num_elements(field)) elements.\n",
  "Values = "
)

#######################################################
# Nodal field stuff
#######################################################
"""
"""
struct QuadratureField{T, N, NFields, NQPoints, NElements, Names, Vals} <: AbstractField{T, N, NFields, Names, Vals}
  names::Names
  vals::Vals
end

Base.size(::QuadratureField{T, N, NFields, NQPoints, NElements, Names, Vals}) where {
  T, N, NFields, NQPoints, NElements, Names, Vals <: AbstractArray{T, 1}
} = (NQPoints, NElements)

# Base.getindex(field::QuadratureField{T, N, NFields, NQPoints, NElements, Names, Vals}, q::Int, e::Int) where {
#   T, N, NFields, NQPoints, NElements, Names, Vals <: AbstractArray{T, 1}
# } = getindex(field.vals, (e - 1) * NQPoints + q)

function Base.getindex(field::QuadratureField{T, N, NF, NQ, NE, Names, Vals}, q::Int, e::Int) where {
  T, N, NF, NQ, NE, Names, Vals <: AbstractArray{T, 1}
} 
  @assert q > 0   "Field index out of range in vectorized getindex"
  @assert e > 0   "Field index out of range in vectorized getindex"
  @assert q <= NQ "Field index out of range in vectorized getindex $q"
  @assert e <= NE "Field index out of range in vectorized getindex"
  getindex(field.vals, (e - 1) * NQ + q)
end

function Base.setindex!(field::QuadratureField{T, N, NF, NQ, NE, Names, Vals}, v, q::Int, e::Int) where {
  T, N, NF, NQ, NE, Names, Vals <: AbstractArray{T, 1}
} 

  @assert q > 0   "Field index out of range in vectorized getindex"
  @assert e > 0   "Field index out of range in vectorized getindex"
  @assert q <= NQ "Field index out of range in vectorized getindex"
  @assert e <= NE "Field index out of range in vectorized getindex"
  setindex!(field.vals, v, (e - 1) * NQ + q)
end

Base.axes(::QuadratureField{T, N, NF, NQ, NE, Names, Vals}, ::Val{1}) where {
  T, N, NF, NQ, NE, Names, Vals <: AbstractArray{T, 1}
} = Base.OneTo(NQ)

Base.axes(::QuadratureField{T, N, NF, NQ, NE, Names, Vals}, ::Val{2}) where {
  T, N, NF, NQ, NE, Names, Vals <: AbstractArray{T, 1}
} = Base.OneTo(NE)

function Base.axes(u::QuadratureField{T, N, NF, NQ, NE, Names, Vals}, n::Int) where {
  T, N, NF, NQ, NE, Names, Vals <: AbstractArray{T, 1}
} 
  @assert n > 0  "Dimension index out of range in vectorized axes"
  @assert n <= 2 "Dimension index out of range in vectorized axes"
  axes(u, Val(n))
end

"""
"""
function QuadratureField{NF, NQ, NE, A, T}(
  ::UndefInitializer, names
) where {NF, NQ, NE, A <:AbstractArray, T}

  if T <: Number
    if NF != 1
      field_number_error(QuadratureField, T, NF)
    end
  elseif T <: Union{MArray, SArray}
    if NF != length(T)
      # TODO add tensor types
      field_number_error(QuadratureField, T, NF)
    end
  else
    field_type_error(QuadratureField, T, Union{<:Number, <:Union{MArray, SArray}})
  end

  if A <: AbstractVector
    # NOTE in this case we are allowing for a long multi dof vector
    # e.g. convert a 3 x N matrix to a 3N vector
    vals = A{T}(undef, NQ * NE)
  else
    vals = A{T}(undef, NQ, NE)
  end

  
  return QuadratureField{eltype(vals), ndims(vals), NF, NQ, NE, typeof(names), typeof(vals)}(names, vals)
end

function zeros!(field::QuadratureField)
  for e in axes(field, 2)
    for q in axes(field, 1)
      field[q, e] = zero(eltype(field))
    end
  end
end

"""
"""
function Base.zeros(::Type{QuadratureField{NFields, NQ, NElements, A, T}}, name) where {NFields, NQ, NElements, A <: AbstractArray, T}
  field = QuadratureField{NFields, NQ, NElements, A, T}(undef, name)
  zeros!(field)
  # field .= zero(T)
  return field
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
