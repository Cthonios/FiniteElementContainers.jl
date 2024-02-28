"""
$(TYPEDEF)
"""
const Connectivity{T, N, NN, NE, Vals}           = ElementField{T, N, NN, NE, Vals}
"""
$(TYPEDEF)
"""
const SimpleConnectivity{T, N, NN, NE, Vals}     = SimpleElementField{T, N, NN, NE, Vals}
"""
$(TYPEDEF)
"""
const VectorizedConnectivity{T, N, NN, NE, Vals} = VectorizedElementField{T, N, NN, NE, Vals}

function SimpleConnectivity{NN, NE}(vals::M) where {NN, NE, M <: AbstractArray{<:Integer, 2}}
  # weirdly need to import base here, can't find eltype(::Matrix{Int64})
  return SimpleConnectivity{Base.eltype(vals), 2, NN, NE, typeof(vals)}(vals)
end

function SimpleConnectivity{NN, NE, Matrix, T}(::UndefInitializer) where {NN, NE, T <: Number}
  vals = Matrix{T}(undef, NN, NE)
  return SimpleConnectivity{T, 2, NN, NE, typeof(vals)}(vals)
end

"""
$(TYPEDSIGNATURES)
"""
connectivity(conn::SimpleConnectivity) = conn.vals
"""
$(TYPEDSIGNATURES)
"""
connectivity(conn::SimpleConnectivity, e::Int) = @views conn.vals[:, e] # TODO maybe a duplicate view here in some cases

###################################################################################

function VectorizedConnectivity{NN, NE}(vals::M) where {NN, NE, M <: Matrix{<:Integer}}
  new_vals = vec(vals)
  return VectorizedConnectivity{eltype(new_vals), 2, NN, NE, typeof(new_vals)}(new_vals)
end

function VectorizedConnectivity{NN, NE}(vals::V) where {NN, NE, V <: Vector{<:Integer}}
  return VectorizedConnectivity{eltype(vals), 2, NN, NE, typeof(vals)}(vals)
end

function VectorizedConnectivity{NN, NE, Vector, SVector}(vals::M) where {NN, NE, M <: Matrix{<:Integer}}
  @assert size(vals) == (NN, NE)
  new_vals = reinterpret(SVector{NN, eltype(vals)}, vec(vals)) |> collect
  return VectorizedConnectivity{eltype(new_vals), 1, NN, NE, typeof(new_vals)}(new_vals)
end

function VectorizedConnectivity{NN, NE, Vector, SVector}(vals::V) where {NN, NE, V <: Vector{<:Integer}}
  @assert size(vals) == (NN * NE,)
  new_vals = reinterpret(SVector{NN, eltype(vals)}, vals) |> collect
  return VectorizedConnectivity{eltype(new_vals), 1, NN, NE, typeof(new_vals)}(new_vals)
end

function VectorizedConnectivity{NN, NE, StructArray, SVector}(vals::M) where {NN, NE, M <: Matrix{<:Integer}}
  @assert size(vals) == (NN, NE)
  new_vals = reinterpret(SVector{NN, eltype(vals)}, vec(vals)) |> collect
  new_vals = StructArray(new_vals)
  return VectorizedConnectivity{eltype(new_vals), 1, NN, NE, typeof(new_vals)}(new_vals)
end

"""
$(TYPEDSIGNATURES)
"""
connectivity(conn::VectorizedConnectivity) = conn.vals
"""
$(TYPEDSIGNATURES)
"""
connectivity(conn::VectorizedConnectivity{T, 1, NN, NE, Vals}, e::Int) where {T, NN, NE, Vals} = conn[e]
"""
$(TYPEDSIGNATURES)
"""
connectivity(conn::VectorizedConnectivity{T, 2, NN, NE, Vals}, e::Int) where {T, NN, NE, Vals} = @views conn[:, e] # TODO maybe a duplicate view here in some cases

###################################################################################

Connectivity{NN, NE, Matrix, T}(vals::Matrix{T}) where {NN, NE, T <: Integer} = SimpleConnectivity{NN, NE}(vals)
Connectivity{NN, NE, Vector, T}(vals::Matrix{T}) where {NN, NE, T <: Integer} = VectorizedConnectivity{NN, NE}(vals)
Connectivity{NN, NE, Vector, T}(vals::Vector{T}) where {NN, NE, T <: Integer} = VectorizedConnectivity{NN, NE}(vals)
Connectivity{NN, NE, Vector, SVector}(vals::Matrix{T}) where {NN, NE, T <: Integer} = VectorizedConnectivity{NN, NE, Vector, SVector}(vals)
Connectivity{NN, NE, Vector, SVector}(vals::Vector{T}) where {NN, NE, T <: Integer} = VectorizedConnectivity{NN, NE, Vector, SVector}(vals)
Connectivity{NN, NE, StructArray, SVector}(vals::Matrix{T}) where {NN, NE, T <: Integer} = VectorizedConnectivity{NN, NE, StructArray, SVector}(vals)

###################################################################################
