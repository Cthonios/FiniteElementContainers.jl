abstract type AbstractAssembler end

struct StaticAssembler{Rtype} <: AbstractAssembler
	R::Vector{Rtype}
	K::SparseMatrixCSC{Rtype, Int64}
end

Base.show(io::IO, a::StaticAssembler{Rtype}) where Rtype = 
print(
	io, "StaticAssembler:\n",
	"  Rtype = $Rtype\n",
	"  Size  = $(size(a.K))\n"
)

function number_of_hessian_entries_naive(dof::DofManager)
	n_hessian_entries = 0
  for conn in dof.dof_conns
    n_hessian_entries += length(conn)^2
  end
	return n_hessian_entries
end

function setup_hessian_coordinates!(
  row_coords::Vector{Int64}, # hardcoded due to limitation in SparseArrays 
  col_coords::Vector{Int64}, # hardcoded due to limitation in SparseArrays 
  dof::DofManager
)

  counter = 1
  for conn in dof.dof_conns
    for i in axes(conn)
      for j in axes(conn)
        row_coords[counter] = conn[i]
        col_coords[counter] = conn[j]
        counter += 1
      end
    end
  end
end

function StaticAssembler(dof::DofManager, Rtype::Type{<:AbstractFloat} = Float64)
	n_hessian_entries = number_of_hessian_entries_naive(dof)
  col_coords = Vector{Int64}(undef, n_hessian_entries) # hardcoded due to limitation in SparseArrays 
  row_coords = Vector{Int64}(undef, n_hessian_entries) # hardcoded due to limitation in SparseArrays 
  setup_hessian_coordinates!(col_coords, row_coords, dof)

	R = Vector{Rtype}(undef, length(dof.is_unknown))
  K = sparse(row_coords, col_coords, zeros(Rtype, length(row_coords)))
  return StaticAssembler{Rtype}(R, K)
end

function assemble!(
  assembler::StaticAssembler{Rtype},
  fspaces,
  dof::DofManager{NDof, B, V, S},
  residual_func::Function,
  U::VecOrMat{Rtype}
) where {Rtype, NDof, B, V, S}

  assembler.R .= 0.

  e_global = 1

  for n in axes(fspaces, 1)
    fspace = fspaces[n]

    N      = length(fspace[1, 1].N)
    NxNDof = N * NDof

    for e in axes(fspace, 2)
      U_el = @views SMatrix{NDof, N, Rtype, NxNDof}(U[:, dof.conns[e].conn])

			R_el = zeros(SVector{NxNDof, Rtype})
			for q in axes(fspace, 1)
				R_el = R_el + residual_func(fspace[q, e], U_el)
			end

      conn = dof.dof_conns[e_global]
      for i in axes(conn)
        assembler.R[conn[i]] += R_el[i]
      end

      e_global += 1
    end
  end
end

function assemble!(
  assembler::StaticAssembler{Rtype},
  fspaces,
  dof::DofManager{NDof, B, V, S},
  residual_func::Function,
  stiffness_func::Function,
  U::VecOrMat{Rtype}
) where {Rtype, NDof, B, V, S}

  assembler.R .= 0.
  assembler.K .= 0.

  e_global = 1

  for n in axes(fspaces, 1)
    fspace = fspaces[n]

    N      = length(fspace[1, 1].N)
    NxNDof = N * NDof

    for e in axes(fspace, 2)
      U_el = @views SMatrix{NDof, N, Rtype, NxNDof}(U[:, dof.conns[e].conn])

			R_el = zeros(SVector{NxNDof, Rtype})
			K_el = zeros(SMatrix{NxNDof, NxNDof, Rtype, NxNDof * NxNDof})
			for q in axes(fspace, 1)
				R_el = R_el + residual_func(fspace[q, e], U_el)
				K_el = K_el + stiffness_func(fspace[q, e], U_el)
			end

      conn = dof.dof_conns[e_global]
      for i in axes(conn)
        assembler.R[conn[i]] += R_el[i]
        for j in axes(conn)
          assembler.K[conn[i], conn[j]] += K_el[i, j]
        end
      end

      e_global += 1
    end
  end
end