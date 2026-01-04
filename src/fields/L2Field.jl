struct L2Field{
    T, # Let it be anything to allow for structs
    D <: AbstractVector{T}
}
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

function block_view(field::L2Field, b::Int)
    nfield = field.nfields[b]
    nepe = field.nepes[b]
    nelem = field.nelems[b]
    boffset = field.offsets[b]
    bend = boffset + nfield * nepe * nelem - 1
    return reshape(view(field.data, boffset:bend), nfield, nepe, nelem)
end
