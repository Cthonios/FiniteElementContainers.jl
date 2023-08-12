struct Assembler{Itype, N, NDof, Rtype, NxNDof, L}
  U_els::Vector{SMatrix{N, NDof, Rtype, NxNDof}}
  scratch_Rs::Vector{SVector{NxNDof, Rtype}}
  scratch_Ks::Vector{SMatrix{NxNDof, NxNDof, Rtype, L}}
  R::Vector{Rtype}
  K::SparseMatrixCSC{Rtype, Itype}
end

function Assembler(
  fspace::FunctionSpace{<:Integer, N, D, Rtype, L1}, 
  dof::DofManager{<:Integer, D, Rtype, NDof}
  # dof::DofManager
) where {N, D, Rtype, L1, NDof}

  # NDof = ndofs(dof)
  NxNDof = N * NDof
  L2 = NxNDof * NxNDof

  U_els = Vector{SMatrix{N, NDof, Rtype, NxNDof}}(undef, size(fspace, 2))

  scratch_Rs = Vector{SVector{NxNDof, Rtype}}(undef, size(fspace)[2])
  scratch_Ks = Vector{SMatrix{NxNDof, NxNDof, Rtype, L2}}(undef, size(fspace)[2])

  R = Vector{Rtype}(undef, length(dof.row_coords))
  K = sparse(dof.row_coords, dof.col_coords, zeros(Rtype, length(dof.row_coords)))

  return Assembler{Int64, N, NDof, Rtype, NxNDof, L2}(U_els, scratch_Rs, scratch_Ks, R, K)
end

function calculate_residuals!(
  assembler::Assembler{<:Integer, N, NDof, Rtype, NxNDof, L1},
  fspace::FunctionSpace{<:Integer, N, D, Rtype, L2},
  U_els::Vector{SVector{N, Rtype}},
  func::Function
) where {N, NDof, Rtype, NxNDof, L1, D, L2}

  for e in axes(fspace, 2)
    assembler.scratch_Rs[e] = zeros(SVector{NxNDof, Rtype})
    for q in axes(fspace, 1)
      assembler.scratch_Rs[e] += func(fspace[q, e], U_els[e])
    end
  end
end

function calculate_tangents!(
  assembler::Assembler{<:Integer, N, NDof, Rtype, NxNDof, L1},
  fspace::FunctionSpace{<:Integer, N, D, Rtype, L2},
  U_els::Vector{SVector{N, Rtype}},
  func::Function
) where {N, NDof, Rtype, NxNDof, L1, D, L2}

  for e in axes(fspace, 2)
    assembler.scratch_Ks[e] = zeros(SMatrix{NxNDof, NxNDof, Rtype, L1})
    for q in axes(fspace, 1)
      assembler.scratch_Ks[e] += func(fspace[q, e], U_els[e])
    end
  end
end

function update_scratch!(
  assembler::Assembler{<:Integer, N, NDof, Rtype, NxNDof, L1},
  fspace::FunctionSpace{<:Integer, N, D, Rtype, L2},
  residual_func::Function,
  tangent_func::Function,
  U::Vector{Rtype}
) where {N, NDof, Rtype, NxNDof, L1, D, L2}

  for e in axes(fspace, 2)
    assembler.U_els[e]      = @views SMatrix{N, NDof, Rtype, NxNDof}(U[fspace.conn[:, e]])
    assembler.scratch_Rs[e] = zeros(SVector{NxNDof, Rtype})
    assembler.scratch_Ks[e] = zeros(SMatrix{NxNDof, NxNDof, Rtype, L1})
    for q in axes(fspace, 1)
      assembler.scratch_Rs[e] += residual_func(fspace[q, e], assembler.U_els[e])
      assembler.scratch_Ks[e] += tangent_func(fspace[q, e], assembler.U_els[e])
    end
  end
end

function reset!(assembler::Assembler{Itype, N, NDof, Rtype, NxNDof, L1}) where {Itype, N, NDof, Rtype, NxNDof, L1}
  assembler.R .= 0.0
  assembler.K .= 0.0
end

function assemble!(
  assembler::Assembler{<:Integer, N, NDof, Rtype, NxNDof, L1},
  fspace::FunctionSpace{<:Integer, N, D, Rtype, L2},
  dof::DofManager{<:Integer, D, Rtype, NDof}
) where {N, NDof, Rtype, NxNDof, L1, D, L2}

  conn_ids = @views dof.ids[fspace.conn]
  for e in axes(assembler.scratch_Rs, 1)
    for i in axes(fspace.conn, 1)
      assembler.R[conn_ids[i, e]] += assembler.scratch_Rs[e][i]
      for j in axes(fspace.conn, 1)
        # assembler.K[conn_ids[i, e], conn_ids[i, e]] += assembler.scratch_Ks[e][i, j]
        assembler.K[conn_ids[i, e], conn_ids[j, e]] += assembler.scratch_Ks[e][i, j]
      end
    end
  end
end

function assemble_residual!(
  assembler::Assembler{<:Integer, N, NDof, Rtype, NxNDof, L1},
  fspace::FunctionSpace{<:Integer, N, D, Rtype, L2},
  dof::DofManager{<:Integer, D, Rtype, NDof}
) where {N, NDof, Rtype, NxNDof, L1, D, L2}

  conn_ids = @views dof.ids[fspace.conn]
  for e in axes(assembler.scratch_Rs, 1)
    for i in axes(fspace.conn, 1)
      assembler.R[conn_ids[i, e]] += assembler.scratch_Rs[e][i]
    end
  end
end
