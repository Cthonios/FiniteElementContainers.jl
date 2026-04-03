struct L2Field{
    T, # Let it be anything to allow for structs
    D <: AbstractVector{T}
} <: AbstractDiscontinuousField{T, D}
    data::D                    # flat storage (CPU or GPU)
    nfields::Vector{Int}
    nepes::Vector{Int} # num nodes, q points, etc.
    nelems::Vector{Int}
    offsets::Vector{Int}
end

function L2Field(arrs::Vector{<:AbstractArray{T, 3}}) where T
    nfields = map(x -> size(x, 1), arrs)
    nepes = map(x -> size(x, 2), arrs)
    nelems = map(x -> size(x, 3), arrs)
    offsets = Vector{eltype(nepes)}(undef, 0)
    offset = 1
    for b in 1:length(nepes)
        push!(offsets, offset)
        offset += nfields[b] * nepes[b] * nelems[b]
    end
    data = mapreduce(vec, vcat, arrs)
    return L2Field(data, nfields, nepes, nelems, offsets)
end

function L2Field(::UndefInitializer, ::Type{T}, nfields::Int, qsizes::Vector{Tuple{Int, Int}}) where T
    arrs = Array{T, 3}[]
    for (nq, ne) in qsizes
        push!(arrs, Array{T, 3}(undef, nfields, nq, ne))
    end
    return L2Field(arrs)
end

function L2Field(::UndefInitializer, ::Type{T}, nfields::Vector{Int}, qsizes::Vector{Tuple{Int, Int}}) where T
    arrs = Array{T, 3}[]
    for (nf, (nq, ne)) in zip(nfields, qsizes)
        push!(arrs, Array{T, 3}(undef, nf, nq, ne))
    end
    return L2Field(arrs)
end

function Adapt.adapt_structure(to, field::L2Field{T, D}) where {T, D}
    data = adapt(to, field.data)
    return L2Field{T, typeof(data)}(
        data,
        field.nfields,
        field.nepes,
        field.nelems,
        field.offsets
    )
end

function Base.show(io::IO, field::L2Field)
    println(io, "L2Field:")
    for b in 1:num_blocks(field)
        nf, nepe, ne = block_size(field, b)
        println(io, "  Block $b:")
        println(io, "    Number of fields               = $nf")
        println(io, "    Number of entities per element = $nepe")
        println(io, "    Number of elements             = $ne")
    end
end

# just to have something to fall back to
Base.getindex(field::L2Field, i::Int) = field.data[i]
Base.IndexStyle(::Type{<:L2Field}) = IndexLinear()
Base.size(field::L2Field) = size(field.data)

function Base.show(io::IO, ::MIME"text/plain", field::L2Field)
    show(io, field)
end

function block_size(field::L2Field, b::Int)
    return (field.nfields[b], field.nepes[b], field.nelems[b])
end

function block_view(field::L2Field, b::Int)
    nfield = field.nfields[b]
    nepe = field.nepes[b]
    nelem = field.nelems[b]
    boffset = field.offsets[b]
    bend = boffset + nfield * nepe * nelem - 1
    return reshape(view(field.data, boffset:bend), nfield, nepe, nelem)
end

function num_blocks(field::L2Field)
    return length(field.nelems)
end
