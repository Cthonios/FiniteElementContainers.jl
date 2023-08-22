struct Assembler{Rtype, Itype}
  R::Vector{Rtype}
  K::SparseMatrixCSC{Rtype, Itype}
  M::SparseMatrixCSC{Rtype, Itype}
end

function Assembler(dof::DofManager{<:Integer, D, Rtype, NDof}) where {D, Rtype, NDof}
  R = Vector{Rtype}(undef, length(dof.row_coords))
  K = sparse(dof.row_coords, dof.col_coords, zeros(Rtype, length(dof.row_coords)))
  M = sparse(dof.row_coords, dof.col_coords, zeros(Rtype, length(dof.row_coords)))
  return Assembler{Rtype, Int64}(R, K, M)
end

function assemble!(
  assembler::Assembler{Rtype, Itype},
  fspace::FunctionSpace{<:Integer, N, D, Rtype, L},
  dof::DofManager{<:Integer, D, Rtype, NDof},
  residual_func::Function,
  stiffness_func::Function,
  U::Vector{Rtype},
) where {Rtype, Itype, N, D, L, NDof}

  NxNDof = N * NDof
  assembler.R .= 0.
  assembler.K .= 0.

  conn_ids = @views dof.ids[fspace.conn]

  L2 = NxNDof * NxNDof
  R_els = Vector{SVector{NxNDof, Rtype}}(undef, size(fspace, 1))
  K_els = Vector{SMatrix{NxNDof, NxNDof, Rtype, L2}}(undef, size(fspace, 1))
  for e in axes(fspace, 2)
    U_el = @views SMatrix{N, NDof, Rtype, NxNDof}(U[fspace.conn[:, e]])

    for q in axes(fspace, 1)
      R_els[q] = residual_func(fspace[q, e], U_el)
      K_els[q] = stiffness_func(fspace[q, e], U_el)
    end

    R_el = sum(R_els)
    K_el = sum(K_els)

    for i in axes(fspace.conn, 1)
      assembler.R[conn_ids[i, e]] += R_el[i]
      for j in axes(fspace.conn, 1)
        assembler.K[conn_ids[i, e], conn_ids[j, e]] += K_el[i, j]
      end
    end
  end
end
