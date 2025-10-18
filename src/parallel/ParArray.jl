# below consts are to help for typing
# for the debug case
const TypeOrVecType{T} = Union{
    T,
    MPIValue{T},
    <:AbstractVector{T}
} where T
const VecOrVecVec{T} = Union{
    <:AbstractVector{T}, 
    <:AbstractVector{<:AbstractVector{T}}
} where T

abstract type AbstractParArrayShard{T, N} <: AbstractArray{T, N} end

Base.IndexStyle(::Type{<:AbstractParArrayShard}) = IndexLinear()
Base.size(a::AbstractParArrayShard) = (length(a.ghost) + length(a.own),)

function Base.getindex(s::AbstractParArrayShard, n::Int)
    @assert n >= 1 && n <= length(s)
    if n <= length(s.own)
        return s.own[n]
    else
        index = n - length(s.own)
        return s.ghost[index]
    end
end

function Base.setindex!(s::AbstractParArrayShard, v, n::Int)
    if n <= length(s.own)
        s.own[n] = v
    else
        index = n - length(s.own)
        s.ghost[index] = v
    end
end

struct ParVectorShard{
    T,
    TV <: AbstractVector{T},
    IV <: AbstractVector{<:Integer}
} <: AbstractParArrayShard{T, 1}
    comm_graph::CommunicationGraph{TV, IV}
    ghost::TV
    own::TV
end

abstract type AbstractParArray{T, N} <: AbstractArray{T, N} end

Base.IndexStyle(::Type{<:AbstractParArray}) = IndexLinear()
Base.axes(a::AbstractParArray) = (Base.axes(a.parts, 1),)
Base.size(a::AbstractParArray) = (length(a.parts),)

Base.getindex(a::AbstractParArray, n::Int) = a.parts[n]
 
struct ParVector{
    T, TV, IV
} <: AbstractParArray{T, 1}
    parts::TypeOrVecType{ParVectorShard{T, TV, IV}}
end

function ParVector(comm_graphs::TypeOrVecType{<:CommunicationGraph})
    parts = map(comm_graphs) do comm_graph
        n_ghost = comm_graph.n_local - comm_graph.n_owned
        n_own = comm_graph.n_owned
        ghost = zeros(Float64, n_ghost)
        own = zeros(Float64, n_own)
        ParVectorShard(comm_graph, ghost, own)
    end
    return ParVector(parts)
end

function scatter_ghosts!(v::ParVector)
    # MPI.Irecv!()
    comm = MPI.COMM_WORLD
    parts = map(v.parts) do part
        for edge in part.comm_graph.edges
            setdatasend!(edge, part)

            dest = edge.rank - 1
            recv_req = MPI.Irecv!(edge.data_recv, comm; source=dest)
            send_req = MPI.Isend(edge.data_send, comm; dest=dest)
            MPI.Waitall([recv_req, send_req])

            for n in axes(edge.data_recv, 1)
                if edge.is_owned_recv[n] == 1 && edge.is_owned_send[n] == 0
                    index = edge.indices[n]
                    part[index] = edge.data_recv[n]
                end
            end
        end
        part
    end
    # copyto!(v.parts, parts)
    # MPI.Barrier(comm)
    new_v = ParVector(parts)
    return new_v
end
