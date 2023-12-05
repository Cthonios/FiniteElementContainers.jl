abstract type AbstractConnectivity{T, N, NNodesPerElement, NElements, Conn} <: AbstractArray{T, N} end

struct Connectivity{T, N, NNodesPerElement, NElements, Conn} <: AbstractConnectivity{T, N, NNodesPerElement, NElements, Conn}
  conn::Conn
end

function Connectivity{NN, NE}(conn::Matrix{<:Integer}, block_index::Int) where {NN, NE}
  conn = ElementField{size(conn, 1), size(conn, 2)}(Symbol("connectivity_id_$block_index"), conn)
  return Connectivity{eltype(conn), ndims(conn), NN, NE, typeof(conn)}(conn)
end

# function Connectivity{NN, NE}(conn::C, block_index::Int) where {NN, NE, C <: Union{Vector{<:SVector}, StructVector{<:SVector}}}
#   # NNodesPerElement = length(conn[1])
#   conn = ElementField{eltype(conn)}(Symbol("connectivity_id_$block_index"), conn)
#   # return Connectivity{eltype(conn), ndims(conn), NNodesPerElement, typeof(conn)}(conn)
#   return Connectivity{eltype(conn), ndims(conn), NN, NE, typeof(conn)}(conn)
# end

Base.IndexStyle(::Type{<:Connectivity})   = IndexLinear()
Base.size(conn::Connectivity)             = size(conn.conn)
Base.getindex(conn::Connectivity, i::Int) = getindex(conn.conn, i)
Base.axes(conn::Connectivity)             = axes(conn.conn)

connectivity(conn::Connectivity) = conn.conn.vals

connectivity(conn::Connectivity{T, N, NN, NE, Conn}, e::Int) where {
  T, N, NN, NE, Conn <: ElementField{T, N, NN, NE, Names, <:Matrix} where Names
} = @views conn.conn[:, e] # TODO maybe a duplicate view here in some cases
# } = conn.conn[:, e]

connectivity(conn::Connectivity{T, N, NN, NE, Conn}, e::Int) where {
  T, N, NN, NE, Conn <: ElementField{T, N, NN, NE, Names, <:Vector{<:Union{SArray, MArray}}} where Names
} = conn.conn[e]

Base.show(io::IO, conn::Connectivity) = print(io, "\nConnectivity:\n", conn.conn)

num_nodes_per_element(::Connectivity{T, N, NN, NE, Conn}) where {T, N, NN, NE, Conn} = NN
num_elements(::Connectivity{T, N, NN, NE, Conn})          where {T, N, NN, NE, Conn} = NE

field_names(conn::Connectivity) = field_names(conn.conn)