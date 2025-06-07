"""
$(TYPEDEF)
"""
const Connectivity{T, NN, Vals, SymIDMap} = L2ElementField{T, NN, Vals, SymIDMap}

function Connectivity{NN, NE}(vals::M) where {NN, NE, M <: AbstractArray{<:Integer, 2}}
  @assert size(vals) == (NN, NE)
  new_vals = vec(vals)
  syms = map(x -> Symbol(:node_, x), 1:NN)
  nt = NamedTuple{tuple(syms...)}(1:NN)
  return Connectivity{eltype(M), NN, typeof(new_vals), nt}(new_vals)
end

function Connectivity{NN, NE}(vals::V) where {NN, NE, V <: AbstractArray{<:Integer, 1}}
  @assert size(vals) == (NN * NE,)
  syms = map(x -> Symbol(:node_, x), 1:NN)
  nt = NamedTuple{tuple(syms...)}(1:NN)
  return Connectivity{eltype(V), NN, typeof(vals), nt}(vals)
end

"""
$(TYPEDSIGNATURES)
"""
Connectivity{Tup}(vals) where Tup = Connectivity{Tup[1], Tup[2]}(vals)

"""
$(TYPEDSIGNATURES)
"""
connectivity(conn::Connectivity) = conn.vals
"""
$(TYPEDSIGNATURES)
"""
connectivity(conn::Connectivity, e::Int) = @views conn[:, e] # TODO maybe a duplicate view here in some cases
