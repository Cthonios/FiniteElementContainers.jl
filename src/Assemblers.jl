abstract type AbstractAssembler end

struct StaticAssembler{Rtype} <: AbstractAssembler
	R::Vector{Rtype}
	K::SparseMatrixCSC{Rtype, Int64}
end

struct DynamicAssembler{Rtype} <: AbstractAssembler
  R::Vector{Rtype}
  K::SparseMatrixCSC{Rtype, Int64}
  M::SparseMatrixCSC{Rtype, Int64}
end

Base.show(io::IO, a::A) where A <: AbstractAssembler = 
print(
	io, "$(typeof(a)):\n",
	"  Rtype = $(eltype(a.R))\n",
	"  Size  = $(size(a.K))\n"
)

function number_of_hessian_entries_naive(fspaces::F) where {F <: AbstractArray{<:FunctionSpace}}
	n_hessian_entries = 0
  # for conn in dof.dof_conns
  #   n_hessian_entries += length(conn)^2
  # end
  for fspace in fspaces
    # n_hessian_entries += connectivity(fspace) |> length
    # n_hessian_entries += connectivity(fspace) |> dof_connectivity |> length
    conn = connectivity(fspace)
    for e in axes(conn)
      n_hessian_entries += length(dof_connectivity(conn, e))^2
    end
  end
	return n_hessian_entries
end

function setup_hessian_coordinates!(
  row_coords::Vector{Int64}, # hardcoded due to limitation in SparseArrays 
  col_coords::Vector{Int64}, # hardcoded due to limitation in SparseArrays 
  # dof::DofManager,
  fspaces::F
) where F <: AbstractArray{<:FunctionSpace}

  counter = 1
  # for conn in dof.dof_conns
  for fspace in fspaces
    conn = connectivity(fspace)
    # for i in axes(conn)
    #   for j in axes(conn)
    #     row_coords[counter] = conn[i]
    #     col_coords[counter] = conn[j]
    #     counter += 1
    #   end
    # end
    for e in axes(conn)
      el_conn = dof_connectivity(conn, e)
      for i in axes(el_conn, 1)
        for j in axes(el_conn, 1)
          # @show el_conn
          # @show el_conn[i]
          row_coords[counter] = el_conn[i]
          col_coords[counter] = el_conn[j]
          counter += 1
        end
      end
    end
  end
end

# function StaticAssembler(dof::DofManager, Rtype::Type{<:AbstractFloat} = Float64)
function StaticAssembler(
  dof::DofManager, fspaces::F, Rtype::Type{<:AbstractFloat} = Float64
) where F <: AbstractArray{<:FunctionSpace}
	n_hessian_entries = number_of_hessian_entries_naive(fspaces)
  col_coords = Vector{Int64}(undef, n_hessian_entries) # hardcoded due to limitation in SparseArrays 
  row_coords = Vector{Int64}(undef, n_hessian_entries) # hardcoded due to limitation in SparseArrays 
  setup_hessian_coordinates!(col_coords, row_coords, fspaces)

	R = Vector{Rtype}(undef, length(dof.is_unknown))
  K = sparse(row_coords, col_coords, zeros(Rtype, length(row_coords)))
  return StaticAssembler{Rtype}(R, K)
end

# function DynamicAssembler(dof::DofManager, Rtype::Type{<:AbstractFloat} = Float64)
function DynamicAssembler(
  dof::DofManager, fspaces::F, Rtype::Type{<:AbstractFloat} = Float64
) where F <: AbstractArray{<:FunctionSpace}
	n_hessian_entries = number_of_hessian_entries_naive(fspaces)
  col_coords = Vector{Int64}(undef, n_hessian_entries) # hardcoded due to limitation in SparseArrays 
  row_coords = Vector{Int64}(undef, n_hessian_entries) # hardcoded due to limitation in SparseArrays 
  setup_hessian_coordinates!(col_coords, row_coords, fspaces)

	R = Vector{Rtype}(undef, length(dof.is_unknown))
  K = sparse(row_coords, col_coords, zeros(Rtype, length(row_coords)))
  M = sparse(row_coords, col_coords, zeros(Rtype, length(row_coords)))
  return DynamicAssembler{Rtype}(R, K, M)
end

function assemble!(
  assembler::A,
  fspaces,
  dof::DofManager{NDof, B, V, Rtype},
  residual_func::Function,
  U::VecOrMat{Rtype}
) where {A <: AbstractAssembler, NDof, B, V, Rtype}

  assembler.R .= 0.

  # e_global = 1

  for n in axes(fspaces, 1)
    fspace = fspaces[n]
    conn   = connectivity(fspace)
    N      = length(fspace[1, 1].N)
    NxNDof = N * NDof

    for e in axes(fspace, 2)
      # U_el = @views SMatrix{NDof, N, Rtype, NxNDof}(U[:, dof.conns[e].conn])
      U_el = @views SMatrix{NDof, N, Rtype, NxNDof}(U[:, element_connectivity(conn, e)])

			R_el = zeros(SVector{NxNDof, Rtype})
			for q in axes(fspace, 1)
				R_el = R_el + residual_func(fspace[q, e], U_el)
			end

      # conn = dof.dof_conns[e_global]
      el_conn = dof_connectivity(conn, e)
      for i in axes(el_conn)
        assembler.R[el_conn[i]] += R_el[i]
      end

      # e_global += 1
    end
  end
end

function assemble!(
  assembler::StaticAssembler{Rtype},
  fspaces,
  # dof,
  dof::DofManager{NDof, B, V, Rtype},
  residual_func::Function,
  stiffness_func::Function,
  U::VecOrMat{Rtype}
) where {NDof, B, V, Rtype}

  assembler.R .= 0.
  assembler.K .= 0.

  # e_global = 1

  for n in axes(fspaces, 1)
    fspace = fspaces[n]
    conn   = connectivity(fspace)
    N      = length(fspace[1, 1].N)
    NxNDof = N * NDof

    for e in axes(fspace, 2)
      # U_el = @views SMatrix{NDof, N, Rtype, NxNDof}(U[:, dof.conns[e].conn])
      U_el = @views SMatrix{NDof, N, Rtype, NxNDof}(U[:, element_connectivity(conn, e)])

			R_el = zeros(SVector{NxNDof, Rtype})
      # R_el = zeros()
			K_el = zeros(SMatrix{NxNDof, NxNDof, Rtype, NxNDof * NxNDof})
			for q in axes(fspace, 1)
				R_el = R_el + residual_func(fspace[q, e], U_el)
				K_el = K_el + stiffness_func(fspace[q, e], U_el)
			end

      # conn = dof.dof_conns[e_global]
      el_conn = dof_connectivity(conn, e)
      for i in axes(el_conn)
        assembler.R[el_conn[i]] += R_el[i]
        for j in axes(el_conn)
          assembler.K[el_conn[i], el_conn[j]] += K_el[i, j]
        end
      end

      # e_global += 1
    end
  end
end

function assemble!(
  assembler::DynamicAssembler{Rtype},
  fspaces,
  # dof,
  dof::DofManager{NDof, B, V, Rtype},
  residual_func::Function,
  stiffness_func::Function,
  mass_func,
  U::VecOrMat{Rtype}
) where {NDof, B, V, Rtype}

  assembler.R .= 0.
  assembler.K .= 0.
  assembler.M .= 0.

  e_global = 1

  for n in axes(fspaces, 1)
    fspace = fspaces[n]

    N      = length(fspace[1, 1].N)
    NxNDof = N * NDof

    for e in axes(fspace, 2)
      U_el = @views SMatrix{NDof, N, Rtype, NxNDof}(U[:, dof.conns[e].conn])

			R_el = zeros(SVector{NxNDof, Rtype})
			K_el = zeros(SMatrix{NxNDof, NxNDof, Rtype, NxNDof * NxNDof})
      M_el = zeros(SMatrix{NxNDof, NxNDof, Rtype, NxNDof * NxNDof})
			for q in axes(fspace, 1)
				R_el = R_el + residual_func(fspace[q, e], U_el)
				K_el = K_el + stiffness_func(fspace[q, e], U_el)
        M_el = M_el + mass_func(fspace[q, e], U_el)
			end

      conn = dof.dof_conns[e_global]
      for i in axes(conn)
        assembler.R[conn[i]] += R_el[i]
        for j in axes(conn)
          assembler.K[conn[i], conn[j]] += K_el[i, j]
          assembler.M[conn[i], conn[j]] += M_el[i, j]
        end
      end

      e_global += 1
    end
  end
end