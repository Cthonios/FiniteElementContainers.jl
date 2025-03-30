"""
$(TYPEDEF)
Thin wrapper that subtypes ```AbstractArray``` and serves
as the base ```Field``` type
"""
abstract type AbstractField{T, N, NF, Vals, SymIDMap} <: AbstractArray{T, N} end

"""
$(TYPEDSIGNATURES)
"""
function KA.get_backend(field::AbstractField)
  return KA.get_backend(field.vals)
end

"""
$(TYPEDSIGNATURES)
"""
Base.eltype(::Type{AbstractField{T, N, NF, Vals, SymIDMap}}) where {T, N, NF, Vals, SymIDMap} = T

"""
$(TYPEDSIGNATURES)
"""
Base.names(::AbstractField{T, N, NF, Vals, SymIDMap}) where {T, N, NF, Vals, SymIDMap} = keys(SymIDMap)

"""
$(TYPEDSIGNATURES)
"""
num_fields(::AbstractField{T, N, NF, Vals, SymIDMap}) where {T, N, NF, Vals, SymIDMap} = NF

"""
$(TYPEDSIGNATURES)
"""
_sym_id_map(::AbstractField{T, N, NF, Vals, SymIDMap}, sym::Symbol) where {T, N, NF, Vals, SymIDMap} = getproperty(SymIDMap, sym)


# actual implementations
# include("ElementField.jl")
include("H1Field.jl")
include("L2ElementField.jl")
include("L2QuadratureField.jl")

# some specialization
include("Connectivity.jl")
