abstract type AbstractBlockProperties{V} end

struct ConstantBlockProperties{
    V <: AbstractArray{<:Number, 1}
} <: AbstractBlockProperties{V}
    props::V
end

struct VariableBlockProperties{
    V <: AbstractArray{<:Number, 1}
} <: AbstractBlockProperties{V}
    props::V
end

struct Properties{
    T <: Number,
    D <: AbstractVector{T}
} <: AbstractDiscontinuousField{T, D}
    data::D
    constant_block::Vector{Bool}
    nprops::Vector{Int}
    nepes::Vector{Int}
    nelems::Vector{Int}
    offsets::Vector{Int}
    prop_names::Vector{Vector{String}}
end

# really basic case constructor where everything is constant
# and there's a default method for all physics...
# this is probably fragile though
function Properties(physics)
    data = Float64[] # TODO eventually type
    constant_block = Bool[]
    nprops = Int[]
    nepes = Int[]
    nelems = Int[]
    offsets = Int[]
    offset = 1
    prop_names = Vector{String}[]
    for physics in physics
        props = create_properties(physics)
        append!(data, props)
        push!(constant_block, true)
        push!(nprops, length(props))
        # for this case we're just going to make em 1
        push!(nepes, 1)
        push!(nelems, 1)
        push!(offsets, offset)
        push!(prop_names, map(x -> "property_$x", 1:length(props)))
        offset += length(props)
    end
    return Properties(data, constant_block, nprops, nepes, nelems, offsets, prop_names)
end

# case where everything is variable and we have a default constructor
function Properties(fspace::FunctionSpace, physics)
    @assert length(physics)
    data = Float64[]
    constant_block = Bool[]
    nprops = Int[]
    nepes = Int[]
    nelems = Int[]
    offsets = Int[]
    offset = 1
    prop_names = Vector{String}[]
    @assert false "Finish me"
end

function Adapt.adapt_structure(to, props::Properties{T, D}) where {T, D}
    data = adapt(to, props.data)
    return Properties{T, typeof(data)}(
        data,
        field.constant_block,
        field.nprops,
        field.nepes,
        field.nelems,
        field.offsets
    )
end

Base.getindex(field::Properties, i::Int) = field.data[i]
Base.IndexStyle(::Type{<:Properties}) = IndexLinear()
Base.size(field::Properties) = size(field.data)

function Base.show(io::IO, props::Properties)
    println(io, "Properties:")
    for b in 1:num_blocks(props)
        nf, nepe, ne = block_size(props, b)
        println(io, "  Block $b:")
        if props.constant_block[b]
            block_props = block_view(props, b)
            for (name, prop) in zip(props.prop_names[b], block_props.props)
                println("    $name => $prop")
            end
        else
            println(io, "    Number of properties           = $nf")
            println(io, "    Number of entities per element = $nepe")
            println(io, "    Number of elements             = $ne")
            println(io, "    Property names:")
            for name in props.prop_names
                println(io, "      $name")
            end
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", field::Properties)
    show(io, field)
end

function block_size(field::Properties, b::Int)
    return (field.nprops[b], field.nepes[b], field.nelems[b])
end

function block_view(props::Properties, b::Int)
    nprops = props.nprops[b]
    boffset = props.offsets[b]
    if props.constant_block[b]
        bend = boffset + nprops - 1
        return ConstantBlockProperties(view(props.data, boffset:bend))
    else
        nepe = field.nepes[b]
        nelem = field.nelems[b]
        bend = boffset + nfield * nepe * nelem - 1
        return VariableBlockProperties(reshape(view(field.data, boffset:bend, nprops, nepe, nelem)))
    end
end

function num_blocks(props::Properties)
    return length(props.nelems)
end
