######################################################################################################
# Abstract type for all fields
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
Base.IndexStyle(::Type{<:AbstractField}) = IndexLinear()
"""
$(TYPEDSIGNATURES)
"""
Base.eltype(::AbstractField{T, N, D}) where {T, N, D} = T
"""
$(TYPEDSIGNATURES)
"""
Base.fill!(field::AbstractField{T, N, D}, v::T) where {T, N, D} = fill!(field.data, v)
"""
$(TYPEDSIGNATURES)
"""
Base.getindex(field::AbstractField, n::Int) = getindex(field.data, n)
"""
$(TYPEDSIGNATURES)
"""
function Base.setindex!(field::AbstractField{T, N, D}, v::T, n::Int) where {T, N, D}
    setindex!(field.data, v, n)
    return nothing
end
"""
$(TYPEDSIGNATURES)
"""
Base.unique(field::AbstractField) = unique(field.data)
"""
$(TYPEDSIGNATURES)
"""
KA.get_backend(field::AbstractField) = KA.get_backend(field.data)

######################################################################################################
# Abstract type for all continuous fields e.g. H1, Hdiv, Hcurl
######################################################################################################
abstract type AbstractContinuousField{T, D <: AbstractArray{T, 1}, NF} <: AbstractField{T, 2, D} end

# minimal abstractarray interface methods below

function Base.axes(field::AbstractContinuousField{T, D, NF}) where {T, D, NF}
    NN = length(field) ÷ NF
    return (Base.OneTo(NF), Base.OneTo(NN))
end

function Base.getindex(field::AbstractContinuousField, d::Int, n::Int)
    @assert d > 0 && d <= num_fields(field)
    @assert n > 0 && n <= num_entities(field)
    return getindex(field.data, (n - 1) * num_fields(field) + d)
end

function Base.resize!(field::AbstractContinuousField{T, D, NF}, n::Int) where {T, D, NF}
    resize!(field.data, NF * n)
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

######################################################################################################
# Abstract type for all block like fields
######################################################################################################
abstract type AbstractBlockField{T, D <: AbstractArray{T, 1}} <: AbstractField{T, 1, D} end

Base.size(field::AbstractBlockField) = size(field.data)

function block_sizes(field::AbstractBlockField)
    return block_size.((field,), 1:num_blocks(field))
end

function num_blocks(field::AbstractBlockField)
    return field.nblocks
end

function num_elements(field::AbstractBlockField)
    return reduce(+, field.nelems)
end

# NOT GPU safe
function num_elements(field::AbstractBlockField{T, D}, b::Int) where {T, D <: Vector{T}}
    return field.nelems[b]
end

# NOT GPU safe
function num_entities_per_element(field::AbstractBlockField{T, D}, b::Int) where {T, D <: Vector{T}}
    return field.nepes[b]
end

######################################################################################################
# Abstract type for all continuous fields e.g. L2Field, StateVariableField, etc.
######################################################################################################
abstract type AbstractDiscontinuousField{T, D <: AbstractArray{T, 1}} <: AbstractBlockField{T, D} end

# need to implement num_fields method
# function num_fields end

function Base.show(io::IO, field::AbstractDiscontinuousField)
    println(io, "$(typeof(field)):")
    for b in 1:num_blocks(field)
        nf, nepe, ne = block_size(field, b)
        println(io, "  Block $b:")
        println(io, "    Number of fields               = $nf")
        println(io, "    Number of entities per element = $nepe")
        println(io, "    Number of elements             = $ne")
    end
end

function Base.show(io::IO, ::MIME"text/plain", field::AbstractDiscontinuousField)
    show(io, field)
end

function block_size(field::AbstractDiscontinuousField, b::Int)
    return (num_fields(field, b), field.nepes[b], field.nelems[b])
end

function block_view(field::AbstractDiscontinuousField, b::Int)
    nfield = num_fields(field, b)
    nepe = field.nepes[b]
    nelem = field.nelems[b]
    boffset = field.offsets[b]
    bend = boffset + nfield * nepe * nelem - 1
    return reshape(view(field.data, boffset:bend), nfield, nepe, nelem)
end

######################################################################################################
# Connectivity
######################################################################################################
"""
$(TYPEDEF)
"""
struct Connectivity{
    T <: Integer, 
    D <: AbstractVector{T}
} <: AbstractBlockField{T, D}
    data::D
    nblocks::T
    nepes::Vector{T}
    nelems::Vector{T}
    offsets::Vector{T}

    function Connectivity(mats::Vector{<:AbstractMatrix{<:Integer}})
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

function block_size(conn::Connectivity, b::Int)
    return (conn.nepes[b], conn.nelems[b])
end

# NOT GPU safe
function connectivity(conn::Connectivity{T, D}, b::Int) where {T, D <: Vector{T}}
    nepe = conn.nepes[b]
    nelem = conn.nelems[b]
    boffset = conn.offsets[b]
    return reshape(view(conn.data, boffset:boffset + nepe * nelem - 1), nepe, nelem)
end

# NOT GPU safe
function connectivity(conn::Connectivity{T, D}, e::Int, b::Int) where {T, D <: Vector{T}}
    nepe = conn.nepes[b]
    boffset = conn.offsets[b]
    start = boffset + nepe * (e - 1)
    finish = boffset + nepe * e - 1
    return view(conn.data, start:finish)
end

# GPU safe
@inline function connectivity(ref_fe::ReferenceFE, conn_data, e::Int, boffset::Int)
    NNPE = ReferenceFiniteElements.num_cell_dofs(ref_fe)
    base = boffset + (e - 1) * NNPE
    data = ntuple(i -> conn_data[base + i - 1], NNPE)
    return SVector{NNPE, Int}(data)
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

######################################################################################################
# Attempt at full GPU Connectivity
######################################################################################################
"""
$(TYPEDEF)
"""
struct Connectivity_v2{
    T <: Integer, 
    D <: AbstractVector{T}
} <: AbstractBlockField{T, D}
    data::D
    nblocks::T
    nepes::D
    nelems::D
    offsets::D
end

function Connectivity_v2(conn::Connectivity)
    return Connectivity_v2(conn.data, conn.nblocks, conn.nepes, conn.nelems, conn.offsets)
end

function Adapt.adapt_structure(to, conn::Connectivity_v2)
    return Connectivity_v2(
        adapt(to, conn.data),
        conn.nblocks,
        adapt(to, conn.nepes),
        adapt(to, conn.nelems),
        adapt(to, conn.offsets)
    )
end

@inline function connectivity(conn::Connectivity_v2, n::Int, e::Int, b::Int)
    idx = conn.offsets[b] + conn.nepes[b] * (e - 1) + n - 1
    return conn.data[idx]
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

function Adapt.adapt_structure(to, field::H1Field{T, D, NF}) where {T, D, NF}
    data = adapt(to, field.data)
    return H1Field{T, typeof(data), NF}(data)
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

    function HcurlField{T, D, NF}(data::D) where {T, D, NF}
        new{T, D, NF}(data)
    end

    function HcurlField{T, D, NF}(data::AbstractMatrix{T}) where {T, D, NF}
        data = vec(data)
        return HcurlField{T, D, NF}(data)
    end

    function HcurlField(data::M) where M <: AbstractMatrix
        NF = size(data, 1)
        data = vec(data)
        return HcurlField{eltype(data), typeof(data), NF}(data)
    end
end

function Adapt.adapt_structure(to, field::HcurlField{T, D, NF}) where {T, D, NF}
    data = adapt(to, field.data)
    return HcurlField{T, typeof(data), NF}(data)
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

    function HdivField{T, D, NF}(data::D) where {T, D, NF}
        new{T, D, NF}(data)
    end

    function HdivField{T, D, NF}(data::AbstractMatrix{T}) where {T, D, NF}
        data = vec(data)
        return HdivField{T, D, NF}(data)
    end

    function HdivField(data::M) where M <: AbstractMatrix
        NF = size(data, 1)
        data = vec(data)
        return HdivField{eltype(data), typeof(data), NF}(data)
    end
end

function Adapt.adapt_structure(to, field::HdivField{T, D, NF}) where {T, D, NF}
    data = adapt(to, field.data)
    return HdivField{T, typeof(data), NF}(data)
end

######################################################################################################
# L2Field
######################################################################################################
struct L2Field{
    T, # Let it be anything to allow for structs
    D  <: AbstractVector{T},
    NF
} <: AbstractDiscontinuousField{T, D}
    data::D              # flat storage (CPU or GPU)
    nblocks::Int
    nepes::Vector{Int}   # num nodes, q points, etc.
    nelems::Vector{Int}
    offsets::Vector{Int}

    function L2Field{T, D, NF}(data, nblocks, nepes, nelems, offsets) where {T, D, NF}
        new{T, D, NF}(data, nblocks, nepes, nelems, offsets)
    end

    function L2Field(arrs::Vector{<:AbstractArray{T, 3}}) where T
        nfields = map(x -> size(x, 1), arrs)
        @assert all(isequal(nfields[1]), nfields)
        nfields = nfields[1]
        nepes = map(x -> size(x, 2), arrs)
        nelems = map(x -> size(x, 3), arrs)
        offsets = Vector{eltype(nepes)}(undef, 0)
        offset = 1
        for b in 1:length(nepes)
            push!(offsets, offset)
            offset += nfields * nepes[b] * nelems[b]
        end
        data = mapreduce(vec, vcat, arrs)
        return L2Field{T, typeof(data), nfields}(data, length(nepes), nepes, nelems, offsets)
    end

    function L2Field(::UndefInitializer, ::Type{T}, nfields::Int, qsizes::Vector{Tuple{Int, Int}}) where T
        arrs = Array{T, 3}[]
        for (nq, ne) in qsizes
            push!(arrs, Array{T, 3}(undef, nfields, nq, ne))
        end
        return L2Field(arrs)
    end

    function L2Field{T, D, NF}(::UndefInitializer, qsizes::Vector{Tuple{Int, Int}}) where {T, D, NF}
        arrs = Array{T, 3}[]
        for (nq, ne) in qsizes
            push!(arrs, Array{T, 3}(undef, NF, nq, ne))
        end
        nepes = map(x -> size(x, 2), arrs)
        nelems = map(x -> size(x, 3), arrs)
        offsets = Vector{eltype(nepes)}(undef, 0)
        offset = 1
        for b in 1:length(nepes)
            push!(offsets, offset)
            offset += NF * nepes[b] * nelems[b]
        end
        data = mapreduce(vec, vcat, arrs)
        return L2Field{T, typeof(data), NF}(data, length(nepes), nepes, nelems, offsets)
    end
end

function Adapt.adapt_structure(to, field::L2Field{T, D, NF}) where {T, D, NF}
    data = adapt(to, field.data)
    return L2Field{T, typeof(data), NF}(
        data,
        field.nblocks,
        field.nepes,
        field.nelems,
        field.offsets
    )
end

function num_fields(::L2Field{T, D, NF}, b::Int) where {T, D, NF}
    return NF
end

######################################################################################################
# PropertyField
######################################################################################################
const PROPS_CONST = -1
const PROPS_ELEMS = -2

struct PropertyField{
    T <: Number,
    D <: AbstractVector{T},
    I <: AbstractVector{Int}
} <: AbstractDiscontinuousField{T, D}
    data::D
    isblockconstant::I
    nblocks::Int
    nepes::I
    nelems::I
    offsets::I

    function PropertyField{T, D, I}(data, isblockconstant, nblocks, nepes, nelems, offsets) where {T, D, I}
        new{T, D, I}(data, isblockconstant, nblocks, nepes, nelems, offsets)
    end

    # One entry per block.  A vector-like entry means the properties are
    # constant across that block; a matrix-like entry means one column per
    # element.  The two may be mixed freely.
    function PropertyField(arrs::AbstractVector)
        isempty(arrs) && throw(ArgumentError(
            "PropertyField needs at least one block of properties, got none"))

        blocks = map(_property_block, arrs)
        T = promote_type(map(eltype, blocks)...)
        blocks = map(x -> convert(AbstractArray{T}, x), blocks)

        nblocks = length(blocks)
        isblockconstant = Vector{Int}(undef, nblocks)
        nepes           = Vector{Int}(undef, nblocks)
        nelems          = Vector{Int}(undef, nblocks)
        offsets         = Vector{Int}(undef, nblocks)

        offset = 1
        for (n, x) in enumerate(blocks)
            elementwise = x isa AbstractMatrix
            isblockconstant[n] = elementwise ? PROPS_ELEMS : PROPS_CONST
            nepes[n]           = elementwise ? size(x, 1) : length(x)
            nelems[n]          = elementwise ? size(x, 2) : -1
            offsets[n]         = offset
            offset            += length(x)
        end

        data = Vector{T}(undef, offset - 1)
        i = 1
        for x in blocks
            n = length(x)
            copyto!(data, i, vec(x), 1, n)
            i += n
        end

        return PropertyField{T, typeof(data), typeof(isblockconstant)}(
            data, isblockconstant, nblocks, nepes, nelems, offsets
        )
    end
end

# Normalize one block's properties to a dense array we own.  This deliberately
# accepts any AbstractVector/AbstractMatrix rather than Vector/Matrix: an
# `SVector` is what a `create_properties` implementation naturally returns, and
# rejecting it left downstream packages with a bare MethodError naming an
# internal constructor.
_property_block(x::AbstractVector{<:Number}) = collect(x)
_property_block(x::AbstractMatrix{<:Number}) = collect(x)
_property_block(x) = throw(ArgumentError(
    "each block's properties must be an AbstractVector of numbers (constant " *
    "across the block) or an AbstractMatrix of numbers with one column per " *
    "element (element-level properties); got $(typeof(x))"))

function Adapt.adapt_structure(to, field::PropertyField)
    data = adapt(to, field.data)
    isblockconstant = adapt(to, field.isblockconstant)
    return PropertyField{eltype(field), typeof(data), typeof(isblockconstant)}(
        data,
        isblockconstant,
        field.nblocks,
        adapt(to, field.nepes),
        adapt(to, field.nelems),
        adapt(to, field.offsets)
    )
end

function num_fields(field::PropertyField, b::Int)
    return field.nepes[b]
end

function properties(field::PropertyField, e::Int, b::Int)
    @assert 1 <= b && b <= field.nblocks
    offset = field.offsets[b]
    nfields = num_fields(field, b)
    if field.isblockconstant[b] == PROPS_CONST
        start = offset
    elseif field.isblockconstant[b] == PROPS_ELEMS
        @assert 1 <= e && e <= field.nelems[b]
        start = offset + nfields * (e - 1)
    end
    return PropertyFieldView(field.data, start, nfields)
end

struct PropertyFieldView{T, D <: AbstractVector{T}} <: AbstractVector{T}
    data::D
    start::Int
    len::Int
end

Base.@propagate_inbounds function Base.getindex(v::PropertyFieldView, i::Int)
    @boundscheck checkbounds(v, i)
    return @inbounds v.data[v.start + i - 1]
end
Base.IndexStyle(::Type{<:PropertyFieldView}) = IndexLinear()
Base.length(v::PropertyFieldView) = v.len
Base.size(v::PropertyFieldView) = (v.len,)

######################################################################################################
# StateVariableField
######################################################################################################
struct StateVariableField{
    T, # Let it be anything to allow for structs
    D <: AbstractVector{T},
    I <: AbstractVector{Int}
} <: AbstractDiscontinuousField{T, D}
    data::D                    # flat storage (CPU or GPU)
    nblocks::Int
    nfields::I
    nepes::I # num nodes, q points, etc.
    nelems::I
    offsets::I

    function StateVariableField{T, D, I}(data, nblocks, nfields, nepes, nelems, offsets) where {T, D, I}
        new{T, D, I}(data, nblocks, nfields, nepes, nelems, offsets)
    end

    function StateVariableField(arrs::Vector{<:AbstractArray{T, 3}}) where T
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
        return StateVariableField{T, typeof(data), typeof(nepes)}(data, length(nepes), nfields, nepes, nelems, offsets)
    end

    function StateVariableField(::UndefInitializer, ::Type{T}, nfields::Int, qsizes::Vector{Tuple{Int, Int}}) where T
        arrs = Array{T, 3}[]
        for (nq, ne) in qsizes
            push!(arrs, Array{T, 3}(undef, nfields, nq, ne))
        end
        return StateVariableField(arrs)
    end

    function StateVariableField(::UndefInitializer, ::Type{T}, nfields::Vector{Int}, qsizes::Vector{Tuple{Int, Int}}) where T
        arrs = Array{T, 3}[]
        for (nf, (nq, ne)) in zip(nfields, qsizes)
            push!(arrs, Array{T, 3}(undef, nf, nq, ne))
        end
        return StateVariableField(arrs)
    end
end

function Adapt.adapt_structure(to, field::StateVariableField{T, D, I}) where {T, D, I}
    data = adapt(to, field.data)
    nfields = adapt(to, field.nfields)
    return StateVariableField{T, typeof(data), typeof(nfields)}(
        data,
        field.nblocks,
        field.nfields,
        adapt(to, field.nepes),
        adapt(to, field.nelems),
        adapt(to, field.offsets)
    )
end

function Base.resize!(field::StateVariableField, block_sizes::Vector{Tuple{Int, Int, Int}})
    n_blocks = field.nblocks
    offset = 1
    for n in 1:n_blocks
        field.nfields[n] = block_sizes[n][1]
        field.nepes[n] = block_sizes[n][2]
        field.nelems[n] = block_sizes[n][3]
        field.offsets[n] = offset
        offset += prod(block_sizes[n])
    end
end

function num_fields(field::StateVariableField, b::Int)
    return field.nfields[b]
end

function state_variables(field::StateVariableField, q::Int, e::Int, b::Int)
    @assert 1 <= q <= field.nepes[b]
    @assert 1 <= e <= field.nelems[b]
    offset  = field.offsets[b]
    nfields = field.nfields[b]
    nqs     = field.nepes[b]
    start   = offset + nfields * (q - 1) + nfields * nqs * (e - 1)
    return StateVariableFieldView(field.data, start, nfields)
end

struct StateVariableFieldView{T, D <: AbstractVector{T}} <: AbstractVector{T}
    data::D
    start::Int
    len::Int
end

Base.@propagate_inbounds function Base.getindex(v::StateVariableFieldView, i::Int)
    @boundscheck checkbounds(v, i)
    return @inbounds v.data[v.start + i - 1]
end
Base.IndexStyle(::Type{<:StateVariableFieldView}) = IndexLinear()
Base.length(v::StateVariableFieldView) = v.len
Base.@propagate_inbounds function Base.setindex!(v::StateVariableFieldView{T, D}, val::T, i::Int) where {T, D}
    @boundscheck checkbounds(v, i)
    @inbounds v.data[v.start + i - 1] = val
    return nothing
end
Base.size(v::StateVariableFieldView) = (v.len,)
