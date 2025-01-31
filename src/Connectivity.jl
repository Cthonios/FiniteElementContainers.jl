"""
$(TYPEDEF)
"""
const Connectivity{T, NN, Vals} = ElementField{T, NN, Vals}

function Connectivity{NN, NE}(vals::M) where {NN, NE, M <: AbstractArray{<:Integer, 2}}
  @assert size(vals) == (NN, NE)
  new_vals = vec(vals)
  return Connectivity{eltype(M), NN, typeof(new_vals)}(new_vals)
end

function Connectivity{NN, NE}(vals::V) where {NN, NE, V <: AbstractArray{<:Integer, 1}}
  @assert size(vals) == (NN * NE,)
  return Connectivity{eltype(V), NN, typeof(vals)}(vals)
end

"""
$(TYPEDSIGNATURES)
"""
Connectivity{Tup}(vals) where Tup = Connectivity{Tup[1], Tup[2]}(vals)

function Connectivity{Tup, T}(::UndefInitializer) where {Tup, T}
  NN, NE = Tup
  vals = Vector{T}(undef, NN * NE)
  return Connectivity{T, NN, typeof(vals)}(vals)
end

"""
$(TYPEDSIGNATURES)
"""
connectivity(conn::Connectivity) = conn.vals
"""
$(TYPEDSIGNATURES)
"""
connectivity(conn::Connectivity, e::Int) = @views conn[:, e] # TODO maybe a duplicate view here in some cases
