function number_of_hessian_entries_naive(dof::DofManager{NDof}) where NDof
	n_hessian_entries = 0
	for conn in dof.conns
		n_nodes_per_elem, n_elem = size(conn)
		n_hessian_entries += n_elem * n_nodes_per_elem^2
	end

	return n_hessian_entries
end

function setup_hessian_coordinates!(
  row_coords::Vector{Int64}, # hardcoded due to limitation in SparseArrays 
  col_coords::Vector{Int64}, # hardcoded due to limitation in SparseArrays 
  dof::DofManager{NDof}
) where NDof

  counter = 1

	for conn in values(dof.conns)
		for e in axes(conn, 2)
			ids_e = @views conn[:, e]

      for i in axes(ids_e, 1)
        for j in axes(ids_e, 1)
          row_coords[counter] = ids_e[i]
          col_coords[counter] = ids_e[j]
          counter = counter + 1
        end
      end
		end
	end
end

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

function StaticAssembler(dof::DofManager, Rtype::Type{<:AbstractFloat} = Float64)
	n_hessian_entries = number_of_hessian_entries_naive(dof)
  col_coords = Vector{Int64}(undef, n_hessian_entries) # hardcoded due to limitation in SparseArrays 
  row_coords = Vector{Int64}(undef, n_hessian_entries) # hardcoded due to limitation in SparseArrays 
  setup_hessian_coordinates!(col_coords, row_coords, dof)

  # R = Vector{Rtype}(undef, length(row_coords))
	R = Vector{Rtype}(undef, length(dof.is_unknown))
  K = sparse(row_coords, col_coords, zeros(Rtype, length(row_coords)))
  return StaticAssembler{Rtype}(R, K)
end

function assemble!(
	assembler::StaticAssembler{Rtype},
	fspaces,
	dof::DofManager{NDof},
	residual_func::Function,
	stiffness_func::Function,
	U::VecOrMat{Rtype}
) where {Rtype, NDof}

	assembler.R .= 0.
	assembler.K .= 0.

	for n in axes(fspaces, 1)
		fspace = fspaces[n]


		N = length(fspace[1, 1].N)
		
		# _, N_shape, _, _ = fspace[1, 1]
		# N = length(N_shape)


		NxNDof = N * NDof
		for e in axes(fspace, 2)
			U_el = @views SMatrix{N, NDof, Rtype, NxNDof}(U[:, dof.conns[n][:, e]])

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

################################################################

function assemble!(
	assembler::StaticAssembler{Rtype},
	fspaces,
	dof::DofManager{NDof},
	residual_func::Function,
	stiffness_func::Function,
	U::VecOrMat{Rtype},
	block_colors::Vector{Dict{Int64, Vector{Int64}}}
) where {Rtype, NDof}

	assembler.R .= 0.
	assembler.K .= 0.

	for n in axes(fspaces, 1)
		fspace = fspaces[n]
		colors = block_colors[n]
		N = length(fspace[1, 1].N)
		NxNDof = N * NDof

		for color in keys(colors)
			Threads.@threads for e in colors[color]
				U_el = @views SMatrix{N, NDof, Rtype, NxNDof}(U[:, dof.conns[n][:, e]])
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
end

################################################################

function assemble!(
	assembler::StaticAssembler{Rtype},
	fspace::FunctionSpace3{N, D, Rtype, L},
	dof::DofManager{NDof},
	residual_func::Function,
	stiffness_func::Function,
	U::VecOrMat{Rtype},
	colors::Vector{Int64},
	fspace_index::Int
) where {Rtype, N, D, L, NDof}

	# NDof = size(U, 1)
	NxNDof = N * NDof

	

	@avxt for c in eachindex(colors)
	# @turbo for c in eachindex(colors)
	# for c in eachindex(colors)
		e = colors[c]

		# U_el = zeros(SMatrix{N, NDof, Rtype, NxNDof})
		# R_el = zeros(SVector{NxNDof, Rtype})
		# K_el = zeros(SMatrix{NxNDof, NxNDof, Rtype, NxNDof * NxNDof})
		U_el = zeros(MArray, N, NDof)
		R_el = zeros(MArray, NxNDof)
		K_el = zeros(MArray, NxNDof, NxNDof)
		for n in 1:N, n_dof in 1:NDof
			U_el[n, n_dof] = U[n_dof, dof.conns[fspace_index][n, e]]
		end

		for q in 1:fspace.q_offset
			R_q = residual_func(fspace[q, e], U_el)
			K_q = stiffness_func(fspace[q, e], U_el)
			
			for n in 1:N
				for n_dof in 1:NDof
					k = NDof * (n - 1) + n_dof
					R_el[k] = R_el[k] + R_q[n_dof, n]

				end
			end
			# R_el = R_el + residual_func(fspace[q, e], U_el)
			# K_el = K_el + stiffness_func(fspace[q, e], U_el)
		end

		# actual assembly here
		# for i in axes(dof.conns[fspace_index], 1)
		# 	# assembler.R[dof.conns[fspace_index][i, e]] += R_el[i]
		# 	for j in axes(dof.conns[fspace_index], 1)
		# 		# assembler.K[dof.conns[fspace_index][i, e], dof.conns[fspace_index][j, e]] += K_el[i, j]
		# 	end
		# end
	end
end

# function assemble!(
# 	assembler::StaticAssembler{Rtype},
# 	fspaces::Vector{FunctionSpace3{N, D, Rtype, L}},
# 	dof::DofManager{NDof},
# 	residual_func::Function,
# 	stiffness_func::Function,
# 	U::VecOrMat{Rtype},
# 	block_colors::Vector{Vector{Vector{Int64}}}
# ) where {Rtype, NDof, N, D, L}

# 	assembler.R .= 0.
# 	assembler.K .= 0.
# 	for n in axes(fspaces, 1)
# 		fspace = fspaces[n]
# 		colors = block_colors[n]
# 		for c in 1:length(colors)
# 			assemble!(assembler, fspace, dof, residual_func, stiffness_func, U, colors[c], n)
# 		end
# 	end
# end

function assemble!(
	assembler::StaticAssembler{Rtype},
	fspaces::Vector{FunctionSpace3{N, D, Rtype, L}},
	dof::DofManager{NDof},
	residual_func::Function,
	stiffness_func::Function,
	U::VecOrMat{Rtype},
	block_colors::Vector{Vector{Vector{Int64}}}
) where {Rtype, NDof, N, D, L}

	assembler.R .= 0.
	assembler.K .= 0.
	NxNDof = N * NDof

	# U_el_buffers = [zeros(SMatrix{N, NDof, Rtype, NxNDof}) for _ in 1:Threads.nthreads()]
	# R_el_buffers = [zeros(SVector{NxNDof, Rtype}) for _ in 1:Threads.nthreads()]
	# K_el_buffers = [zeros(SMatrix{NxNDof, NxNDof, Rtype, NxNDof * NxNDof}) for _ in 1:Threads.nthreads()]
	# K_q_cache_buffers = [zeros(SMatrix{NxNDof, NxNDof, Rtype, NxNDof * NxNDof}) for _ in 1:Threads.nthreads()]

	for n in axes(fspaces, 1)
		fspace = fspaces[n]
		colors = block_colors[n]
		conns = dof.conns[n]
		# display(colors)
		
		for els in colors

			# threads loop here
			Threads.@threads for e in els
				# need to allocate at each thread, or make a buffer later on
				# U_el = U_el_buffers[Threads.threadid()]
				# # U_el .= 0.
				# R_el = R_el_buffers[Threads.threadid()]
				# R_el .= 0.
				# K_el = K_el_buffers[Threads.threadid()]
				# K_el .= 0.

				# fill the element level array
				# for n_node in 1:N, n_dof in 1:NDof
				# 	U_el[n_node, n_dof] = @views U[n_dof, dof.conns[n][n_node, e]]
				# end

				U_el = @views SMatrix{N, NDof, Rtype, NxNDof}(U[:, conns[:, e]])
				R_el = zeros(SVector{NxNDof, Rtype})
				K_el = zeros(SMatrix{NxNDof, NxNDof, Rtype, NxNDof * NxNDof})

				# caclulate residual and jacobian
				for q in 1:fspace.q_offset
					R_q = residual_func(fspace[q, e], U_el)
					K_q = stiffness_func(fspace[q, e], U_el)
					# K_q = K_q_cache_buffers[Threads.threadid()]
					# K_q .= 0.
					# K_q = stiffness_func(K_q, fspace[q, e], U_el)

					R_el += R_q
					K_el += K_q
				end

				# actual assembly here
				for i in axes(conns, 1)
					assembler.R[conns[i, e]] += R_el[i]
					for j in axes(conns, 1)
						assembler.K[conns[i, e], conns[j, e]] += K_el[i, j]
					end
				end

				nothing
			end
		end
	end
end


####################################
# Buffers
####################################
# hardcoded to scalar equation
struct ThreadedAssemblyCache{N, D, Rtype, L}
	u_q::Vector{Rtype}
	∇u_q::Vector{MVector{D, Rtype}}
	R_q::Vector{MVector{N, Rtype}}
	K_q::Vector{MMatrix{N, N, Rtype, L}}
end
Base.length(t::ThreadedAssemblyCache) = length(t.u_q)

function ThreadedAssemblyCache(n_threads::Int, N::Int, D::Int, Rtype::Type = Float64)
	u_q = zeros(Rtype, n_threads)
	∇u_q = [zeros(MVector{D, Rtype}) for _ in 1:n_threads]
	R_q = [zeros(MVector{N, Rtype}) for _ in 1:n_threads]
	K_q = [zeros(MMatrix{N, N, Rtype, N * N}) for _ in 1:n_threads]
	return ThreadedAssemblyCache{N, D, Rtype, N * N}(u_q, ∇u_q, R_q, K_q)
end