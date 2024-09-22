# NOTE not completely functional and clean yet
"""
$(TYPEDEF)
$(TYPEDFIELDS)
NFS and NES are tuples of the number of fields
and elements respectively
"""
struct ComponentArrayElementField{
  T, NFS, NES, Vals <: ComponentArray
}
  vals::Vals
end

"""
$(TYPEDSIGNATURES)
"""
num_elements(::ComponentArrayElementField{T, NF, NE, Vals}) where {T, NF, NE, Vals} = NE
"""
$(TYPEDSIGNATURES)
"""
num_fields(::ComponentArrayElementField{T, NF, NE, Vals}) where {T, NF, NE, Vals} = NF

"""
$(TYPEDSIGNATURES)
"""
function ComponentArrayElementField{NFS, NES, T}(::UndefInitializer, names) where {NFS, NES, T}
  vals = map((f, e) -> Array{T, 2}(undef, f, e), NFS, NES)
  vals = ComponentArray(NamedTuple{names}(vals))
  return ComponentArrayElementField{T, NFS, NES, typeof(vals)}(vals)
end

function Base.show(io::IO, field::A) where A <: ComponentArrayElementField
  println(io, "ComponentArrayElementField")
  for (n, block) in enumerate(valkeys(field.vals))
    NF, NE = size(field, n)
    println(io, "  $(Symbol(block))")
    println(io, "    Number of fields            = $NF")
    println(io, "    Number of elements          = $NE")
  end
end

"""
$(TYPEDSIGNATURES)
"""
Base.getindex(field::A, i) where A <: ComponentArrayElementField = getindex(field.vals, i)

"""
$(TYPEDSIGNATURES)
TODO has a few allocations to clean up
"""
function Base.setindex!(field::A, val, block, n, e) where A <: ComponentArrayElementField 
  setindex!(view(field.vals, block), val, n, e)
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
Base.size(field::A) where A <: ComponentArrayElementField = size(field.vals)
"""
$(TYPEDSIGNATURES)
"""
function Base.size(field::A, i::Int) where A <: ComponentArrayElementField
  NFS = num_fields(field)
  NES = num_elements(field)
  return NFS[i], NES[i]
end
