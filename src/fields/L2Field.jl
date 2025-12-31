struct L2Field{
    T, D
} <: AbstractField{T, 2, D, 1}
    data::D                    # flat storage (CPU or GPU)
    block_offsets::Vector{Int}   # host-side
    block_sizes::Vector{Int}     # NB per block
    block_elems::Vector{Int}     # NE per block
end

function L2Field(block_mats::Vector{<:AbstractMatrix})
    T = eltype(block_mats[1])

    nb = length(block_mats)
    block_sizes  = Vector{Int}(undef, nb)
    block_elems  = Vector{Int}(undef, nb)
    block_offsets = Vector{Int}(undef, nb)

    total = 0
    for b in 1:nb
        block_sizes[b] = size(block_mats[b], 1)
        block_elems[b] = size(block_mats[b], 2)
        block_offsets[b] = total + 1
        total += block_sizes[b] * block_elems[b]
    end

    data = Vector{T}(undef, total)

    for b in 1:nb
        NB = block_sizes[b]
        off = block_offsets[b]
        copyto!(data, off, vec(block_mats[b]), 1, NB * block_elems[b])
    end

    return L2Field{T, typeof(data)}(
        data, block_offsets, block_sizes, block_elems
    )
end

function Adapt.adapt_structure(to, field::L2Field{T, D}) where {T, D}
    data = adapt(to, field.data)
    return L2Field{T, typeof(data)}(
        data,
        field.block_offsets,
        field.block_sizes,
        field.block_elems,
    )
end

@inline function element_view(
    data::AbstractVector{T},
    offset::Int,
    NB::Int,
    e::Int,
) where T
    @views data[offset + (e - 1) * NB : offset + e * NB - 1]
end

function element_view(field::L2Field, e, b)
    offset = field.block_offsets[b]
    ND = field.block_sizes[b]
    fi = offset + (e - 1) * ND
    li = offset + e * ND - 1

    return @view field.data[fi:li]
end

function Base.getindex(field::L2Field, d::Int, e::Int, b::Int)
    offset = field.block_offsets[b]
    ND = field.block_sizes[b]
    index = offset + (e - 1) * ND - 1 + d
    return field.data[index]
end

function Base.setindex!(field::L2Field, v, d::Int, e::Int, b::Int)
    offset = field.block_offsets[b]
    ND = field.block_sizes[b]
    index = offset + (e - 1) * ND - 1 + d
    field.data[index] = v
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", field::L2Field)
    println("L2Field output disabled")
    # println("  Size = $(size(field))")
end