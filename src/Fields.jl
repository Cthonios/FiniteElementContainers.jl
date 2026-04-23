######################################################################################################
# Abstract types
######################################################################################################
"""
$(TYPEDEF)
Thin wrapper that subtypes ```AbstractArray``` and serves
as the base ```Field``` type
"""
abstract type AbstractField{T, N, D <: AbstractArray{T, 1}} <: AbstractArray{T, N} end
"""
$(TYPEDSIGNATURES)
"""
Base.fill!(field::AbstractField{T, N, D}, v::T) where {T, N, D} = fill!(field.data, v)
"""
$(TYPEDSIGNATURES)
"""
Base.unique(field::AbstractField) = unique(field.data)
"""
$(TYPEDSIGNATURES)
"""
KA.get_backend(field::AbstractField) = KA.get_backend(field.data)

abstract type AbstractContinuousField{T, D <: AbstractArray{T, 1}, NF} <: AbstractField{T, 2, D} end

function Adapt.adapt_structure(to, field::AbstractContinuousField{T, D, NF}) where {T, D, NF}
    data = adapt(to, field.data)
    type = typeof(field).name.name
    return eval(type){T, typeof(data), NF}(data)
end

# minimal abstractarray interface methods below

function Base.axes(field::AbstractContinuousField{T, D, NF}) where {T, D, NF}
    NN = length(field) ÷ NF
    return (Base.OneTo(NF), Base.OneTo(NN))
end

function Base.getindex(field::AbstractContinuousField, n::Int)
  return getindex(field.data, n)
end

function Base.getindex(field::AbstractContinuousField, d::Int, n::Int)
    @assert d > 0 && d <= num_fields(field)
    @assert n > 0 && n <= num_entities(field)
    return getindex(field.data, (n - 1) * num_fields(field) + d)
end

function Base.IndexStyle(::Type{<:AbstractContinuousField}) 
    return IndexLinear()
end

function Base.setindex!(field::AbstractContinuousField{T, D, NF}, v::T, n::Int) where {T, D, NF}
    setindex!(field.data, v, n)
    return nothing
end 

function Base.setindex!(field::AbstractContinuousField{T, D, NF}, v, d::Int, n::Int) where {T, D, NF}
    @assert d > 0 && d <= num_fields(field)
    @assert n > 0 && n <= num_entities(field)
    setindex!(field.data, v, (n - 1) * num_fields(field) + d)
    return nothing
end

function Base.similar(field::AbstractContinuousField)
  data = similar(field.data)
  return typeof(field)(data)
end

function Base.size(field::AbstractContinuousField{T, D, NF}) where {T, D, NF} 
  NN = length(field.data) ÷ NF
  return (NF, NN)
end

"""
$(TYPEDSIGNATURES)
"""
function num_entities(field::AbstractContinuousField{T, D, NF}) where {T, D, NF}
  return length(field.data) ÷ NF
end
"""
$(TYPEDSIGNATURES)
"""
function num_fields(::AbstractContinuousField{T, D, NF}) where {T, D, NF}
  return NF
end

abstract type AbstractDiscontinuousField{T, D <: AbstractArray{T, 1}} <: AbstractField{T, 1, D} end

######################################################################################################
# Connectivity
######################################################################################################
"""
$(TYPEDEF)
"""
struct Connectivity{
    T <: Integer, 
    D <: AbstractVector{T}
}
    data::D
    nblocks::T
    nepes::Vector{T}
    nelems::Vector{T}
    offsets::Vector{T}
# end

    function Connectivity(mats::Vector{<:AbstractArray{<:Integer, 2}})
        nblocks = length(mats)
        nepes = map(x -> size(x, 1), mats)
        nelems = map(x -> size(x, 2), mats)
        offsets = Vector{eltype(nepes)}(undef, 0)
        offset = 1
        for (nepe, nelem) in zip(nepes, nelems)
            push!(offsets, offset)
            offset += nepe * nelem
        end
        data = mapreduce(vec, vcat, mats)
        # return Connectivity(data, nblocks, nepes, nelems, offsets)
        new{eltype(data), typeof(data)}(data, nblocks, nepes, nelems, offsets)
    end

    function Connectivity(data, nblocks, nepes, nelems, offsets)
        new{eltype(data), typeof(data)}(data, nblocks, nepes, nelems, offsets)
    end

    function Connectivity{T, D}() where {T, D}
        new{T, D}(T[], 0, T[], T[], T[])
    end
end

function Adapt.adapt_structure(to, conn::Connectivity{T, D}) where {T, D}
    return Connectivity(
        adapt(to, conn.data),
        conn.nblocks,
        conn.nepes,
        conn.nelems,
        conn.offsets
    )
end

# NOT GPU safe
function connectivity(conn::Connectivity, b::Int)
    nepe = conn.nepes[b]
    nelem = conn.nelems[b]
    boffset = conn.offsets[b]
    return reshape(view(conn.data, boffset:boffset + nepe * nelem - 1), nepe, nelem)
end

# GPU safe
@inline function connectivity(ref_fe::ReferenceFE, conn_data, e::Int, boffset::Int)
    NNPE = ReferenceFiniteElements.num_cell_dofs(ref_fe)
    base = boffset + (e - 1) * NNPE
    data = ntuple(i -> conn_data[base + i - 1], NNPE)
    return SVector{NNPE, Int}(data)
end

function num_blocks(conn::Connectivity)
    return conn.nblocks
end

function num_elements(conn::Connectivity)
    return sum(conn.nelems)
end

# NOT GPU safe
function num_elements(conn::Connectivity, b::Int)
    return conn.nelems[b]
end

# NOT GPU safe
function num_entities_per_element(conn::Connectivity, b::Int)
    return conn.nepes[b]
end

# GPU safe
@inline function surface_connectivity(ref_fe::ReferenceFE, conn_data, side::Int, e::Int, boffset::Int)
    # Stride through conn_data using the VOLUME element DOF count, not the surface element's.
    # The connectivity array is packed with NNPE_vol entries per element; using NNPE_surf as
    # the stride reads from the wrong position for element e > 1.
    NNPE_vol  = ReferenceFiniteElements.num_cell_dofs(ref_fe)
    face_nodes = ReferenceFiniteElements.boundary_dofs(ref_fe, side)  # 1-based local indices
    NNPE_surf = length(face_nodes)
    base = boffset + (e - 1) * NNPE_vol
    data = ntuple(i -> conn_data[base + face_nodes[i] - 1], NNPE_surf)
    return SVector{NNPE_surf, Int}(data)
end

function unsafe_connectivity(conn::Connectivity, e::Int, b::Int)
    nepe = conn.nepes[b]
    boffset = conn.offsets[b]
    start = boffset + nepe * (e - 1)
    finish = boffset + nepe * e - 1
    return view(conn.data, start:finish)
end

######################################################################################################
# H1Field
######################################################################################################
"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
Implementation of fields that live on nodes.
"""
struct H1Field{T, D, NF} <: AbstractContinuousField{T, D, NF}
    data::D

    function H1Field{T, D, NF}(data::D) where {T, D, NF}
        new{T, D, NF}(data)
    end

    function H1Field{T, D, NF}(data::AbstractMatrix{T}) where {T, D, NF}
        data = vec(data)
        return H1Field{T, D, NF}(data)
    end

    function H1Field(data::M) where M <: AbstractMatrix
        NF = size(data, 1)
        data = vec(data)
        return H1Field{eltype(data), typeof(data), NF}(data)
    end
end

######################################################################################################
# HcurlField
######################################################################################################
"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
Implementation of fields that live in Hdiv spaces.
"""
struct HcurlField{T, D, NF} <: AbstractContinuousField{T, D, NF}
    data::D
end

"""
$(TYPEDSIGNATURES)
"""
function HcurlField(data::M) where M <: AbstractMatrix
    NF = size(data, 1)
    data = vec(data)
    return HcurlField{eltype(data), typeof(data), NF}(data)
end

######################################################################################################
# HdivField
######################################################################################################
"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
Implementation of fields that live in Hdiv spaces.
"""
struct HdivField{T, D, NF} <: AbstractContinuousField{T, D, NF}
    data::D
end

"""
$(TYPEDSIGNATURES)
"""
function HdivField(data::M) where M <: AbstractMatrix
    NF = size(data, 1)
    data = vec(data)
    return HdivField{eltype(data), typeof(data), NF}(data)
end

######################################################################################################
# L2Field
######################################################################################################
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
