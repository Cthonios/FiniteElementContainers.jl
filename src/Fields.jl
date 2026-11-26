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
get_data(field::AbstractField) = field.data
"""
$(TYPEDSIGNATURES)
"""
function set_data!(field::AbstractField{T, N, D}, new_data::D) where {T, N, D}
    field.data .= new_data
end
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
Base.fill!(field::AbstractField{T, N, D}, v::T) where {T, N, D} = fill!(get_data(field), v)
"""
$(TYPEDSIGNATURES)
"""
Base.getindex(field::AbstractField, n::Int) = getindex(get_data(field), n)
"""
$(TYPEDSIGNATURES)
"""
function Base.setindex!(field::AbstractField{T, N, D}, v::T, n::Int) where {T, N, D}
    setindex!(get_data(field), v, n)
    return nothing
end
"""
$(TYPEDSIGNATURES)
"""
Base.unique(field::AbstractField) = unique(get_data(field))
"""
$(TYPEDSIGNATURES)
"""
KA.get_backend(field::AbstractField) = KA.get_backend(get_data(field))

struct FieldAccessException <: Exception
end

function _field_access_exception()
    throw(FieldAccessException())
end

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
    @boundscheck 1 <= d <= num_fields(field) || _field_access_exception()
    @boundscheck 1 <= n <= num_entities(field) || _field_access_exception()
    return getindex(get_data(field), (n - 1) * num_fields(field) + d)
end

function Base.resize!(field::AbstractContinuousField{T, D, NF}, n::Int) where {T, D, NF}
    resize!(get_data(field), NF * n)
    return nothing
end

function Base.setindex!(field::AbstractContinuousField{T, D, NF}, v, d::Int, n::Int) where {T, D, NF}
    @boundscheck 1 <= d <= num_fields(field) || _field_access_exception()
    @boundscheck 1 <= n <= num_entities(field) || _field_access_exception()
    setindex!(get_data(field), v, (n - 1) * num_fields(field) + d)
    return nothing
end

function Base.similar(field::AbstractContinuousField)
  data = similar(get_data(field))
  return typeof(field)(data)
end

function Base.size(field::AbstractContinuousField{T, D, NF}) where {T, D, NF} 
  NN = length(get_data(field)) ÷ NF
  return (NF, NN)
end

"""
$(TYPEDSIGNATURES)
"""
function num_entities(field::AbstractContinuousField{T, D, NF}) where {T, D, NF}
  return length(get_data(field)) ÷ NF
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
struct BlockLayout{I, IV <: AbstractVector{I}}
    nblocks::I
    nepes::IV
    nelems::IV
    offsets::IV

    function BlockLayout(nblocks, nepes, nelems, offsets)
        new{eltype(nepes), typeof(nepes)}(nblocks, nepes, nelems, offsets)
    end

    function BlockLayout{I}() where I <: Integer
        new{I, Vector{I}}(0, I[], I[], I[])
    end
end

Base.eltype(::BlockLayout{I, IV}) where {I, IV} = I

abstract type AbstractBlockField{I, IV <: AbstractVector{I}} <: AbstractField{I, 1, IV} end

Base.size(field::AbstractBlockField) = size(get_data(field))

function block_offset(field::AbstractBlockField, b)
    return field.layout.offsets[b]
end

function block_sizes(field::AbstractBlockField)
    return block_size.((field,), 1:num_blocks(field))
end

function num_blocks(field::AbstractBlockField)
    return field.layout.nblocks
end

function num_elements(field::AbstractBlockField)
    return reduce(+, field.layout.nelems)
end

# NOT GPU safe
function num_elements(field::AbstractBlockField, b::Int)
    return field.layout.nelems[b]
end

# NOT GPU safe
function num_entities_per_element(field::AbstractBlockField, b::Int)
    return field.layout.nepes[b]
end

######################################################################################################
# Abstract type for all continuous fields e.g. L2Field, StateVariableField, etc.
######################################################################################################
abstract type AbstractDiscontinuousField{
    # T, D <: AbstractArray{T, 1}
    I  <: Integer,
    T,
    IV <: AbstractVector{I},
    TV <: AbstractVector{T}
} <: AbstractBlockField{T, TV} end

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
    return (num_fields(field, b), num_entities_per_element(field, b), num_elements(field, b))
end

function block_view(field::AbstractDiscontinuousField, b::Int)
    nfield = num_fields(field, b)
    nepe = num_entities_per_element(field, b)
    nelem = num_elements(field, b)
    boffset = block_offset(field, b)
    bend = boffset + nfield * nepe * nelem - 1
    return reshape(view(get_data(field), boffset:bend), nfield, nepe, nelem)
end

######################################################################################################
# Connectivity
######################################################################################################
"""
$(TYPEDEF)
"""
struct Connectivity{
    I  <: Integer, 
    IV <: AbstractVector{I}
} <: AbstractBlockField{I, IV}
    data::IV
    layout::BlockLayout{I, Vector{I}}

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
        layout = BlockLayout(nblocks, nepes, nelems, offsets)
        new{eltype(data), typeof(data)}(data, layout)
    end

    function Connectivity(data, layout)
        new{eltype(data), typeof(data)}(data, layout)
    end

    function Connectivity{I, IV}() where {I, IV}
        new{I, IV}(T[], BlockLayout{I}())
    end
end

function block_size(conn::Connectivity, b::Int)
    return (conn.nepes[b], conn.nelems[b])
end

# NOT GPU safe
function connectivity(conn::Connectivity{I, IV}, b::Int) where {I, IV <: Vector{I}}
    nepe = num_entities_per_element(conn, b)
    nelem = num_elements(conn, b)
    boffset = block_offset(conn, b)
    return reshape(view(get_data(conn), boffset:boffset + nepe * nelem - 1), nepe, nelem)
end

# NOT GPU safe
function connectivity(conn::Connectivity{I, IV}, e::Int, b::Int) where {I, IV <: Vector{I}}
    nepe = num_entities_per_element(conn, b)
    boffset = block_offset(conn, b)
    start = boffset + nepe * (e - 1)
    finish = boffset + nepe * e - 1
    return view(get_data(conn), start:finish)
end

# GPU safe.  The tuple length is passed as a Val: `ntuple(f, n::Int)` is
# unrolled by Base only for n <= 10, and for a larger element (fifteen nodes)
# it returns a tuple of unknown length, which makes the SVector construction
# and everything after it in a GPU kernel a dynamic call.
@inline function connectivity(ref_fe::ReferenceFE, conn_data, e::Int, boffset::Int)
    NNPE = ReferenceFiniteElements.num_cell_dofs(ref_fe)
    base = boffset + (e - 1) * NNPE
    data = ntuple(i -> conn_data[base + i - 1], Val(NNPE))
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
    data = ntuple(i -> conn_data[base + face_nodes[i] - 1], Val(NNPE_surf))
    return SVector{NNPE_surf, Int}(data)
end

######################################################################################################
# Attempt at full GPU Connectivity
######################################################################################################
"""
$(TYPEDEF)
"""
struct Connectivity_v2{
    I  <: Integer, 
    IV <: AbstractVector{I}
} <: AbstractBlockField{I, IV}
    data::IV
    layout::BlockLayout{I, IV}
end

function Connectivity_v2(conn::Connectivity)
    return Connectivity_v2(get_data(conn), conn.layout)
end

@inline function connectivity(conn::Connectivity_v2, n::Int, e::Int, b::Int)
    idx = block_offset(conn, b) + num_entities_per_element(conn, b) * (e - 1) + n - 1
    return get_data(conn)[idx]
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

######################################################################################################
# L2Field
######################################################################################################
struct L2Field{
    I  <: Integer,
    T,  # Let it be anything to allow for structs
    IV <: AbstractVector{I},
    TV <: AbstractVector{T},
    NF
} <: AbstractDiscontinuousField{I, T, IV, TV}
    data::TV              # flat storage (CPU or GPU)
    layout::BlockLayout{I, Vector{I}}

    function L2Field{I, T, IV, TV, NF}(data, layout) where {I, T, IV, TV, NF}
        new{I, T, IV, TV, NF}(data, layout)
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
        layout = BlockLayout(length(nepes), nepes, nelems, offsets)
        return L2Field{eltype(layout), T, typeof(layout.nelems), typeof(data), nfields}(data, layout)
    end

    function L2Field(::UndefInitializer, ::Type{T}, nfields::Int, qsizes::Vector{Tuple{Int, Int}}) where T
        arrs = Array{T, 3}[]
        for (nq, ne) in qsizes
            push!(arrs, Array{T, 3}(undef, nfields, nq, ne))
        end
        return L2Field(arrs)
    end

    function L2Field{I, T, IV, TV, NF}(::UndefInitializer, qsizes::Vector{Tuple{Int, Int}}) where {I, T, IV, TV, NF}
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
        layout = BlockLayout(length(nepes), nepes, nelems, offsets)
        return L2Field{eltype(layout), T, typeof(layout.nelems), typeof(data), NF}(data, layout)
    end
end

function num_fields(::L2Field{I, T, IV, TV, NF}, b::Int) where {I, T, IV, TV, NF}
    return NF
end

######################################################################################################
# PropertyField
######################################################################################################
const PROPS_CONST = -1
const PROPS_ELEMS = -2
const PropertyBlock{T} = Union{Vector{T}, Matrix{T}}

struct PropertyField{
    I  <: Integer,
    T  <: Number,
    IV <: AbstractVector{I},
    TV <: AbstractVector{T}
} <: AbstractDiscontinuousField{I, T, IV, TV}
    data::TV
    isblockconstant::IV
    layout::BlockLayout{I, IV}

    function PropertyField{I, T, IV, TV}(data, isblockconstant, layout) where {I, T, IV, TV}
        new{I, T, IV, TV}(data, isblockconstant, layout)
    end

    # One entry per block.  A vector-like entry means the properties are
    # constant across that block; a matrix-like entry means one column per
    # element.  The two may be mixed freely.
    function PropertyField(arrs::Vector{<:PropertyBlock{T}}) where {T <: Number}
        isempty(arrs) && throw(ArgumentError("PropertyField needs at least one block of properties"))

        nblocks = length(arrs)

        isblockconstant = Vector{Int}(undef, nblocks)
        nepes = Vector{Int}(undef, nblocks)
        nelems = Vector{Int}(undef, nblocks)
        offsets = Vector{Int}(undef, nblocks)

        offset = 1
        for n in 1:nblocks
            x = arrs[n]
            if x isa Vector{T}
                isblockconstant[n] = PROPS_CONST
                nepes[n] = length(x)
                nelems[n] = -1
            else
                isblockconstant[n] = PROPS_ELEMS
                nepes[n] = size(x, 1)
                nelems[n] = size(x, 2)
            end
            offsets[n] = offset
            offset += length(x)
        end
        data = Vector{T}(undef, offset - 1)
        i = 1
        for n in 1:nblocks
            x = arrs[n]
            nx = length(x)
            copyto!(data, i, vec(x), 1, nx)
            i += nx
        end

        layout = BlockLayout(nblocks, nepes, nelems, offsets)
        return PropertyField{eltype(layout), T, typeof(layout.nelems), typeof(data)}(
            data, isblockconstant, layout
        )
    end
end

# Any other list of blocks: an array literal that mixes a constant block and an
# element-level block (element type `Array{Float64}`, as `_setup_properties`
# builds it from a NamedTuple or a Dict), static arrays, views, or invalid
# input.  Each block is converted to a `Vector` or `Matrix` of one promoted
# element type and the typed constructor above is called; invalid input fails
# with an `ArgumentError` that names the problem.  A trimmed app passes a
# `Vector{<:PropertyBlock{T}}` and does not reach this method.
function PropertyField(arrs::AbstractVector)
    isempty(arrs) && throw(ArgumentError(
        "PropertyField needs at least one block of properties, got none"))
    blocks = map(_property_block, arrs)
    T = promote_type(map(eltype, blocks)...)
    return PropertyField(PropertyBlock{T}[convert(Array{T}, b) for b in blocks])
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

function num_fields(field::PropertyField, b::Int)
    return num_entities_per_element(field, b)
end

function properties(field::PropertyField, e::Int, b::Int)
    @boundscheck 1 <= b <= num_blocks(field) || _field_access_exception()
    offset = block_offset(field, b)
    nfields = num_fields(field, b)
    # `if/elseif` with no `else` leaves `start` undefined on any third value,
    # which surfaces as an `UndefVarError` rather than saying what went wrong.
    # Only two layouts exist, so make the second branch total and assert it.
    if field.isblockconstant[b] == PROPS_CONST
        start = offset
    else
        @boundscheck field.isblockconstant[b] == PROPS_ELEMS || _field_access_exception()
        @boundscheck 1 <= e <= num_elements(field, b) || _field_access_exception()
        start = offset + nfields * (e - 1)
    end
    return PropertyFieldView(get_data(field), start, nfields)
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
    I  <: Integer,
    T, # Let it be anything to allow for structs
    IV <: AbstractVector{I},
    TV <: AbstractVector{T}
} <: AbstractDiscontinuousField{I, T, IV, TV}
    data::TV                    # flat storage (CPU or GPU)
    layout::BlockLayout{I, IV}
    nfields::IV

    function StateVariableField{I, T, IV, TV}(data, layout, nfields) where {I, T, IV, TV}
        new{I, T, IV, TV}(data, layout, nfields)
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
        layout = BlockLayout(length(nepes), nepes, nelems, offsets)
        return StateVariableField{eltype(layout), T, typeof(layout.nelems), typeof(data)}(
            data, layout, nfields
        )
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

function Base.resize!(field::StateVariableField, block_sizes::Vector{Tuple{Int, Int, Int}})
    n_blocks = field.nblocks
    offset = 1
    for n in 1:n_blocks
        field.nfields[n] = block_sizes[n][1]
        field.layout.nepes[n] = block_sizes[n][2]
        field.layout.nelems[n] = block_sizes[n][3]
        field.layout.offsets[n] = offset
        offset += prod(block_sizes[n])
    end
end

function num_fields(field::StateVariableField, b::Int)
    return field.nfields[b]
end

function state_variables(field::StateVariableField, q::Int, e::Int, b::Int)
    @boundscheck 1 <= q <= num_entities_per_element(field, b) || _field_access_exception()
    @boundscheck 1 <= e <= num_elements(field, b) || _field_access_exception()
    offset  = block_offset(field, b)
    nfields = num_fields(field, b)
    nqs     = num_entities_per_element(field, b)
    start   = offset + nfields * (q - 1) + nfields * nqs * (e - 1)
    return StateVariableFieldView(get_data(field), start, nfields)
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
