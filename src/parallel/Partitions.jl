abstract type AbstractShard{T} <: AbstractArray{T, 1} end

Base.IndexStyle(::Type{<:AbstractShard}) = IndexLinear()
Base.length(s::AbstractShard) = length(s.ghost) + length(s.own)
Base.size(s::AbstractShard) = (length(s),)

struct ShardIndexOutOfRange <: Exception
end

_part_index_out_of_range() = throw(ShardIndexOutOfRange())

function _check_index_out_of_range(s::AbstractShard, n::Int)
    if n >= 1 && n <= lb
        return
    else
        _part_index_out_of_range()
    end
end

function Base.getindex(s::AbstractShard, n::Int)
    _check_index_out_of_range(s, n)
    if n <= length(s.own)
        return s.own[n]
    else
        index = n - length(s.own)
        return s.ghost[index]
    end
end

function Base.setindex!(s::AbstractShard, v, n::Int)
    _check_index_out_of_range(s, n)
    if n <= length(s.own)
        s.own[n] = v
    else
        @assert false "Cannot set ghost index via setindex! interface!"
    end
end

# method to set any index as an override to Base.setindex!
# this way we make the user be careful about when to use
# this without have to duplicate own/ghost in a local array
function setlocalindex!(s::AbstractShard, v, n::Int)
    _check_index_out_of_range(s, n)
    if n <= length(s.own)
        s.own[n] = v
    else
        index = n - length(s.own)
        s.ghost[index] = v
    end
end

# write interface for setting ghost elements


struct VectorShard{
    T,
    VT <: AbstractVector{T}
} <: AbstractShard{T}
    ghost::VT
    # ghost_procs::VT
    own::VT
end
