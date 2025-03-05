"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
struct ComponentArrayQuadratureField{
  T, NFS, NQS, NES, Vals <: ComponentArray
} #<: QuadratureField{T, 1, NFS, NQS, NES, Vals}
  vals::Vals
end

"""
$(TYPEDSIGNATURES)
"""
num_elements(::ComponentArrayQuadratureField{T, NF, NQ, NE, Vals}) where {T, NF, NQ, NE, Vals} = NE
"""
$(TYPEDSIGNATURES)
"""
num_fields(::ComponentArrayQuadratureField{T, NF, NQ, NE, Vals}) where {T, NF, NQ, NE, Vals} = NF
"""
$(TYPEDSIGNATURES)
"""
num_q_points(::ComponentArrayQuadratureField{T, NF, NQ, NE, Vals}) where {T, NF, NQ, NE, Vals} = NQ

"""
$(TYPEDSIGNATURES)
"""
function ComponentArrayQuadratureField{NFS, NQS, NES, T}(::UndefInitializer, names) where {NFS, NQS, NES, T}
  vals = map((f, q, e) -> Array{T, 3}(undef, f, q, e), NFS, NQS, NES)
  vals = ComponentArray(NamedTuple{names}(vals))
  return ComponentArrayQuadratureField{T, NFS, NQS, NES, typeof(vals)}(vals)
end

"""
$(TYPEDSIGNATURES)
"""
function Base.show(io::IO, field::A) where A <: ComponentArrayQuadratureField
  println(io, "ComponentArrayQuadratureField")
  for (n, block) in enumerate(valkeys(field.vals))
    NF, NQ, NE = size(field, n)
    println(io, "  $(Symbol(block))")
    println(io, "    Number of fields            = $NF")
    println(io, "    Number of quadrature points = $NQ")
    println(io, "    Number of elements          = $NE")
  end
end

"""
$(TYPEDSIGNATURES)
"""
Base.getindex(field::A, i) where A <: ComponentArrayQuadratureField = getindex(field.vals, i)
"""
$(TYPEDSIGNATURES)
"""
function Base.setindex!(field::A, val, block, n, q, e) where A <: ComponentArrayQuadratureField
  setindex!(view(field.vals, block), val, n, q, e)
  return nothing
end
"""
$(TYPEDSIGNATURES)
"""
Base.size(field::A) where A <: ComponentArrayQuadratureField = size(field.vals)
"""
$(TYPEDSIGNATURES)
"""
function Base.size(field::A, i::Int) where A <: ComponentArrayQuadratureField
  NFS = num_fields(field)
  NQS = num_q_points(field)
  NES = num_elements(field)
  return NFS[i], NQS[i], NES[i]
end
