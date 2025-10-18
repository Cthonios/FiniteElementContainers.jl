abstract type AbstractMPIData{T} <: AbstractArray{T, 1} end

Base.IndexStyle(::Type{<:AbstractMPIData}) = Base.IndexLinear()
Base.getindex(a::AbstractMPIData, n) = a.data[1]
Base.size(a::AbstractMPIData) = size(a.data)

struct MPIValue{T} <: AbstractMPIData{T}
    comm::MPI.Comm
    data::T
end

function MPIValue(data)
    comm = MPI.COMM_WORLD
    return MPIValue(comm, data)
end

function Base.map(f, v::MPIValue)
    new_data = f(v.data)
    return MPIValue(v.comm, new_data)
end

function Base.map(f, v1::MPIValue, v2::MPIValue)
    new_data = f(v1.data, v2.data)
    return MPIValue(v1.comm, new_data)
end

function Base.show(io::IO, v::MPIValue)
    rank = MPI.Comm_rank(v.comm) + 1
    println(io, "MPIValue on rank $rank")
    println(io, v.data)
end

function getdata(v::MPIValue)
    return v.data
end

function setdata(v::MPIValue{T}, data::T) where T
    if ismutable(data)
        if isa(data, AbstractArray)
            copyto!(v.data, data)
        else
            @assert false "Unsupported type setting in setdata in MPIValue"
        end
    else
        @assert false "Can't set immutable data in MPIValue"
    end
end
