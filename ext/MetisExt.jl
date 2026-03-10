module MetisExt

import FiniteElementContainers: SparseMatrixPattern
using Metis

function Metis.partition(pattern::SparseMatrixPattern, nparts::Int)
    Is = convert(Vector{Int32}, pattern.Is)
    Js = convert(Vector{Int32}, pattern.Js)
    Vs = ones(Int8, length(Is))
    n_nodes = length(unique(Is))
    g = sparse(Is, Js, Vs, n_nodes, n_nodes)
    fill!(g.nzval, Int8(1))
    return Metis.partition(Symmetric(g), nparts)
end

end # module
