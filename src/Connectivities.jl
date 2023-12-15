const Connectivity{T, N, NN, NE, Vals}           = ElementField{T, N, NN, NE, Vals}
const SimpleConnectivity{T, N, NN, NE, Vals}     = SimpleElementField{T, N, NN, NE, Vals}
const VectorizedConnectivity{T, N, NN, NE, Vals} = VectorizedElementField{T, N, NN, NE, Vals}

function SimpleConnectivity{NN, NE}(vals::Matrix{<:Integer}) where {NN, NE}
  # weirdly need to import base here, can't find eltype(::Matrix{Int64})
  return SimpleConnectivity{Base.eltype(vals), 2, NN, NE, typeof(vals)}(vals)
end

function SimpleConnectivity{NN, NE, Matrix, T}(::UndefInitializer) where {NN, NE, T <: Number}
  vals = Matrix{T}(undef, NN, NE)
  return SimpleConnectivity{T, 2, NN, NE, typeof(vals)}(vals)
end

connectivity(conn::SimpleConnectivity) = conn.vals
connectivity(conn::SimpleConnectivity, e::Int) = @views conn.vals[:, e] # TODO maybe a duplicate view here in some cases

###################################################################################

function VectorizedConnectivity{NN, NE}(vals::Vector{<:Integer}) where {NN, NE}
  return VectorizedConnectivity{eltype(vals), 2, NN, NE, typeof(vals)}(vals)
end

connectivity(conn::VectorizedConnectivity) = conn.vals
connectivity(conn::VectorizedConnectivity, e::Int) = @views conn[:, e] # TODO maybe a duplicate view here in some cases

###################################################################################

Connectivity{NN, NE, Matrix, T}(vals::Matrix{T}) where {NN, NE, T <: Integer} = SimpleConnectivity{NN, NE}(vals)
Connectivity{NN, NE, Vector, T}(vals::Vector{T}) where {NN, NE, T <: Integer} = VectorizedConnectivity{NN, NE}(vals)

###################################################################################
