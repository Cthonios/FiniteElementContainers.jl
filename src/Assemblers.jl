abstract type AbstractAssembler end

struct StaticAssembler{Rtype, Itype} <: AbstractAssembler
  R::Vector{Rtype}
  K::SparseMatrixCSC{Rtype, Int64}
end

function StaticAssembler(
  dof::DofManager{Itype, Rtype, NDof},
) where {Itype, Rtype, NDof}

  n_hessian_entries = number_of_hessian_entries_naive(dof)
  col_coords = Vector{Itype}(undef, n_hessian_entries)
  row_coords = Vector{Itype}(undef, n_hessian_entries)
  setup_hessian_coordinates!(col_coords, row_coords, dof)

  R = Vector{Rtype}(undef, length(row_coords))
  K = sparse(row_coords, col_coords, zeros(Rtype, length(row_coords)))
  return StaticAssembler{Rtype, Itype}(R, K)
end

function assemble!(
  assembler::StaticAssembler{Rtype, Itype},
  mesh::Mesh,
  fspaces,
  dof::DofManager{<:Integer, Rtype, NDof},
  residual_func::Function,
  U::Vector{Rtype}
) where {Rtype, Itype, NDof}

  assembler.R .= 0.

  for n in axes(fspaces, 1)
    fspace = fspaces[n]
    N = length(fspace[1, 1].N)
    NxNDof = N * NDof

    for e in axes(fspace, 2)
      U_el = @views SMatrix{N, NDof, Rtype, NxNDof}(U[dof.conns[n][:, e]])

      R_el = zeros(SVector{NxNDof, Rtype})
      for q in axes(fspace, 1)
        R_el = R_el + residual_func(fspace[q, e], U_el)
      end

      for i in axes(dof.conns[n], 1)
        assembler.R[dof.conns[n][i, e]] += R_el[i]
      end
    end
  end
end

function assemble!(
  assembler::StaticAssembler{Rtype, Itype},
  mesh::Mesh,
  fspaces,
  dof::DofManager{<:Integer, Rtype, NDof},
  residual_func::Function,
  U::Matrix{Rtype}
) where {Rtype, Itype, NDof}

  assembler.R .= 0.

  for n in axes(fspaces, 1)
    fspace = fspaces[n]
    N = length(fspace[1, 1].N)
    NxNDof = N * NDof

    conn = mesh.blocks[n].conn
    for e in axes(fspace, 2)
      U_el = @views SMatrix{NDof, N, Rtype, NxNDof}(U[:, conn[:, e]])


      R_el = zeros(SVector{NxNDof, Rtype})
      for q in axes(fspace, 1)
        # works
        R_el = R_el + residual_func(fspace[q, e], U_el)

        # maybe something to think about later
        # Rs = vcat(residual_func(fspace[q, e], U_el)...)
        # R_el = R_el + Rs[:]
      end

      for i in axes(dof.conns[n], 1)
        assembler.R[dof.conns[n][i, e]] += R_el[i]
      end
    end
  end
end

function assemble!(
  assembler::StaticAssembler{Rtype, Itype},
  mesh::Mesh,
  fspaces,
  dof::DofManager{<:Integer, Rtype, NDof},
  residual_func::Function,
  stiffness_func::Function,
  U::Vector{Rtype}
) where {Rtype, Itype, NDof}

  assembler.R .= 0.

  # loop over blocks
  for n in axes(fspaces, 1)
    fspace = fspaces[n]
    N = length(fspace[1, 1].N)
    NxNDof = N * NDof

    for e in axes(fspace, 2)
      U_el = @views SMatrix{N, NDof, Rtype, NxNDof}(U[dof.conns[n][:, e]])

      R_el = zeros(SVector{NxNDof, Rtype})
      K_el = zeros(SMatrix{NxNDof, NxNDof, Rtype, NxNDof * NxNDof})
      for q in axes(fspace, 1)
        R_el = R_el + residual_func(fspace[q, e], U_el)
        K_el = K_el + stiffness_func(fspace[q, e], U_el)
      end

      # actual assembly here
      for i in axes(dof.conns[n], 1)
        assembler.R[dof.conns[n][i, e]] += R_el[i]
        for j in axes(dof.conns[n], 1)
          assembler.K[dof.conns[n][i, e], dof.conns[n][j, e]] += K_el[i, j]
        end
      end

    end
  end
end

function assemble!(
  assembler::StaticAssembler{Rtype, Itype},
  mesh::Mesh,
  fspaces,
  dof::DofManager{<:Integer, Rtype, NDof},
  # conns,
  residual_func::Function,
  stiffness_func::Function,
  U::Matrix{Rtype}
) where {Rtype, Itype, NDof}

  assembler.R .= 0.
  assembler.K .= 0.

  # loop over blocks
  for n in axes(fspaces, 1)
    fspace = fspaces[n]
    N = length(fspace[1, 1].N)
    NxNDof = N * NDof
    conn = mesh.blocks[n].conn

    for e in axes(fspace, 2)
      # U_el = @views SMatrix{NDof, N, Rtype, NxNDof}(U[:, conns[n][:, e]])
      U_el = @views SMatrix{NDof, N, Rtype, NxNDof}(U[:, conn[:, e]])

      R_el = zeros(SVector{NxNDof, Rtype})
      K_el = zeros(SMatrix{NxNDof, NxNDof, Rtype, NxNDof * NxNDof})
      for q in axes(fspace, 1)

        # works
        R_el = R_el + residual_func(fspace[q, e], U_el)
        K_el = K_el + stiffness_func(fspace[q, e], U_el)
      end

      # actual assembly here
      for i in axes(dof.conns[n], 1)
        assembler.R[dof.conns[n][i, e]] += R_el[i]
        for j in axes(dof.conns[n], 1)
          assembler.K[dof.conns[n][i, e], dof.conns[n][j, e]] += K_el[i, j]
        end
      end

    end
  end
end
