"""
$(TYPEDEF)
"""
struct Connectivity{
    N, # number of blocks, need to assert that the size of nepes, nelems, offsets never changes (we'll get there)
    T <: Integer, 
    D <: AbstractVector{T}
}
    data::D
    nepes::Vector{T}
    nelems::Vector{T}
    offsets::Vector{T}
end

function Connectivity(mats::Vector{<:AbstractArray{<:Integer, 2}})
    nepes = map(x -> size(x, 1), mats)
    nelems = map(x -> size(x, 2), mats)
    offsets = Vector{eltype(nepes)}(undef, 0)
    offset = 1
    for (nepe, nelem) in zip(nepes, nelems)
        push!(offsets, offset)
        offset += nepe * nelem
    end
    data = mapreduce(vec, vcat, mats)
    return Connectivity{length(nepes), eltype(data), typeof(data)}(data, nepes, nelems, offsets)
end

function Adapt.adapt_structure(to, conn::Connectivity{N, T, D}) where {N, T, D}
    data = adapt(to, conn.data)
    return Connectivity{length(conn.nepes), eltype(data), typeof(data)}(
        data,
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

function num_blocks(conn::Connectivity{N, T, D}) where {N, D, T}
    return N
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
