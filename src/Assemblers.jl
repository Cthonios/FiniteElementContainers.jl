# struct Assembler{Rtype, Itype}
#   R::Vector{Rtype}
#   K::SparseMatrixCSC{Rtype, Itype}
#   M::SparseMatrixCSC{Rtype, Itype}
# end
# issue with Int32 maybe put a pull request in to julia itself

"""
"""
struct Assembler{Rtype, Itype}
  R::Vector{Rtype}
  K::SparseMatrixCSC{Rtype, Int64}
  M::SparseMatrixCSC{Rtype, Int64}
end

"""
"""
function Assembler(dof::DofManager{Itype, Rtype, NDof}) where {Itype, Rtype, NDof}
  R = Vector{Rtype}(undef, length(dof.row_coords))
  K = sparse(dof.row_coords, dof.col_coords, zeros(Rtype, length(dof.row_coords)))
  M = sparse(dof.row_coords, dof.col_coords, zeros(Rtype, length(dof.row_coords)))
  return Assembler{Rtype, Itype}(R, K, M)
end

"""
"""
struct AssemblerCache{NxNDof, Rtype, L}
  R_el::Vector{SVector{NxNDof, Rtype}}
  K_el::Vector{SMatrix{NxNDof, NxNDof, Rtype, L}}
  M_el::Vector{SMatrix{NxNDof, NxNDof, Rtype, L}}
end

"""
"""
function AssemblerCache(
  ::DofManager{Itype, Rtype, NDof},
  fspace::FunctionSpace{Itype, N, D, Rtype, L}
) where {Itype, N, D, Rtype, L, NDof}
  NxNDof = N * NDof
  R_el = Vector{SVector{NxNDof, Rtype}}(undef, size(fspace, 1))
  K_el = Vector{SMatrix{NxNDof, NxNDof, Rtype, NxNDof * NxNDof}}(undef, size(fspace, 1))
  M_el = Vector{SMatrix{NxNDof, NxNDof, Rtype, NxNDof * NxNDof}}(undef, size(fspace, 1))
  return AssemblerCache{NxNDof, Rtype, NxNDof * NxNDof}(R_el, K_el, M_el)
end

struct AssemblerBlock{Itype, N, D, Rtype, L1, NxNDof, L2}
  fspace::FunctionSpace{Itype, N, D, Rtype, L1}
  cache::AssemblerCache{NxNDof, Rtype, L2}
end

function AssemblerBlock(
  dof::DofManager{Itype, Rtype, NDof},
  fspace::FunctionSpace{Itype, N, D, Rtype, L}
) where {Itype, N, D, Rtype, L, NDof}

  cache = AssemblerCache(dof, fspace)
  return AssemblerBlock{Itype, N, D, Rtype, L1, N * NDof, Rtype, L2}(fspace, cache)
end

function assemble!(
  assembler::Assembler{Rtype, Itype},
  assembler_cache::AssemblerCache{NxNDof, Rtype, L1},
  fspace::FunctionSpace{<:Integer, N, D, Rtype, L2},
  dof::DofManager{<:Integer, Rtype, NDof},
  residual_func::Function,
  U::Vector{Rtype},
) where {NxNDof, Rtype, L1, Itype, N, D, L2, NDof}

  assembler.R .= 0.

  conn_ids = @views dof.ids[fspace.conn]

  for e in axes(fspace, 2)
    U_el = @views SMatrix{N, NDof, Rtype, NxNDof}(U[fspace.conn[:, e]])

    for q in axes(fspace, 1)
      assembler_cache.R_el[q] = residual_func(fspace[q, e], U_el)
    end

    R_el = sum(assembler_cache.R_el)

    for i in axes(fspace.conn, 1)
      assembler.R[conn_ids[i, e]] += R_el[i]
    end
  end
end

function assemble!(
  assembler::Assembler{Rtype, Itype},
  assembler_cache::AssemblerCache{NxNDof, Rtype, L1},
  fspace::FunctionSpace{<:Integer, N, D, Rtype, L2},
  dof::DofManager{<:Integer, Rtype, NDof},
  residual_func::Function,
  stiffness_func::Function,
  U::Vector{Rtype},
) where {NxNDof, Rtype, L1, Itype, N, D, L2, NDof}

  assembler.R .= 0.
  assembler.K .= 0.

  conn_ids = @views dof.ids[fspace.conn]

  for e in axes(fspace, 2)
    U_el = @views SMatrix{N, NDof, Rtype, NxNDof}(U[fspace.conn[:, e]])

    for q in axes(fspace, 1)
      assembler_cache.R_el[q] = residual_func(fspace[q, e], U_el)
      assembler_cache.K_el[q] = stiffness_func(fspace[q, e], U_el)
    end

    R_el = sum(assembler_cache.R_el)
    K_el = sum(assembler_cache.K_el)

    for i in axes(fspace.conn, 1)
      assembler.R[conn_ids[i, e]] += R_el[i]
      for j in axes(fspace.conn, 1)
        assembler.K[conn_ids[i, e], conn_ids[j, e]] += K_el[i, j]
      end
    end
  end
end

function assemble!(
  assembler::Assembler{Rtype, Itype},
  assembler_cache::AssemblerCache{NxNDof, Rtype, L1},
  fspace::FunctionSpace{<:Integer, N, D, Rtype, L2},
  dof::DofManager{<:Integer, Rtype, NDof},
  residual_func::Function,
  stiffness_func::Function,
  mass_func::Function,
  U::Vector{Rtype},
) where {NxNDof, Rtype, L1, Itype, N, D, L2, NDof}

  assembler.R .= 0.
  assembler.K .= 0.
  assembler.M .= 0.

  conn_ids = @views dof.ids[fspace.conn]

  for e in axes(fspace, 2)
    U_el = @views SMatrix{N, NDof, Rtype, NxNDof}(U[fspace.conn[:, e]])

    for q in axes(fspace, 1)
      assembler_cache.R_el[q] = residual_func(fspace[q, e], U_el)
      assembler_cache.K_el[q] = stiffness_func(fspace[q, e], U_el)
      assembler_cache.M_el[q] = mass_func(fspace[q, e], U_el)
    end

    R_el = sum(assembler_cache.R_el)
    K_el = sum(assembler_cache.K_el)
    M_el = sum(assembler_cache.M_el)

    for i in axes(fspace.conn, 1)
      assembler.R[conn_ids[i, e]] += R_el[i]
      for j in axes(fspace.conn, 1)
        assembler.K[conn_ids[i, e], conn_ids[j, e]] += K_el[i, j]
        assembler.M[conn_ids[i, e], conn_ids[j, e]] += M_el[i, j]
      end
    end
  end
end
