"""
$(TYPEDEF)
"""
struct Connectivity{T <: Integer, D <: AbstractVector{T}}# <: AbstractArray{T, 1}
    data::D
    nepes::Vector{T}
    nelems::Vector{T}
    offsets::Vector{T}
end

function Connectivity(mats::Vector{<:AbstractArray})
    nepes = map(x -> size(x, 1), mats)
    nelems = map(x -> size(x, 2), mats)
    offsets = Vector{eltype(nepes)}(undef, 0)
    offset = 1
    for (nepe, nelem) in zip(nepes, nelems)
        push!(offsets, offset)
        offset += nepe * nelem
    end
    data = mapreduce(vec, vcat, mats)
    return Connectivity(data, nepes, nelems, offsets)
end

function Adapt.adapt_structure(to, conn::Connectivity{T, D}) where {T, D}
    data = adapt(to, conn.data)
    return Connectivity{T, typeof(data)}(
        data,
        conn.nepes,
        conn.nelems,
        conn.offsets
    )
end

function connectivity(conn::Connectivity, b::Int)
    nepe = conn.nepes[b]
    nelem = conn.nelems[b]
    boffset = conn.offsets[b]
    return reshape(view(conn.data, boffset:boffset + nepe * nelem - 1), nepe, nelem)
end

@inline function connectivity(ref_fe::ReferenceFE, conn_data, e::Int, boffset::Int)
    NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
    base = boffset + (e - 1) * NNPE
    data = ntuple(i -> conn_data[base + i - 1], NNPE)
    return SVector{NNPE, Int}(data)
end

function num_blocks(conn::Connectivity)
    return length(conn.nepes)
end 

function num_elements(conn::Connectivity, b::Int)
    return conn.nelems[b]
end
