struct DofManager{NDof}
	is_unknown::BitMatrix
	unknown_indices::Vector{Int64}
	conns::Vector{Matrix{Int64}}
	dof_conns::Vector{Matrix{Int64}}
end

function DofManager(mesh::Mesh{F, I, B}, n_dofs::Int) where {F, I, B}
	# Note this implementation assumes you are using
  # all of the blocks in the exodus file

	# setup an is unknown array
	# by default everything is unknown and you have to update_bcs! the
	# dof manager downstream
	n_nodes = size(mesh.coords, 2)
	is_unknown = BitArray(undef, n_dofs, n_nodes)
	is_unknown .= 1
	ids = reshape(1:length(is_unknown), n_dofs, n_nodes)
	unknown_indices = ids[is_unknown]

	conns     = Vector{Matrix{Int64}}(undef, length(mesh.blocks))
	dof_conns = Vector{Matrix{Int64}}(undef, length(mesh.blocks))
	for (n, block) in enumerate(mesh.blocks)
		conns[n]     = block.conn
		dof_conns[n] = Matrix{Int64}(undef, n_dofs * size(block.conn, 1), size(block.conn, 2))
		
		for e in axes(block.conn, 2)
			@views dof_conns[n][:, e] = vec(ids[:, block.conn[:, e]])
		end
	end

	return DofManager{n_dofs}(is_unknown, unknown_indices, conns, dof_conns)
end

create_fields(d::DofManager, Rtype::Type{<:AbstractFloat} = Float64) = zeros(Rtype, size(d.is_unknown))
create_unknowns(d::DofManager, Rtype::Type{<:AbstractFloat} = Float64) = zeros(Rtype, sum(d.is_unknown))
dof_ids(d::DofManager) = reshape(1:length(d.is_unknown), size(d.is_unknown))
Base.size(d::DofManager) = size(d.is_unknown)

# allocations mainly from the closures
function update_bcs!(
  U::Matrix{Rtype}, 
  mesh::Mesh{Rtype, I, B},
	dof::DofManager,
  bcs::Vector{<:EssentialBC}
) where {Rtype, I, B}

	dof.is_unknown .= 1
  for bc in bcs
		@inbounds dof.is_unknown[bc.nodes] .= false
    @inbounds U[bc.dof, bc.nodes] .= @views bc.func.(eachcol(mesh.coords[:, bc.nodes]), (0.,))
  end

	new_unknown_indices = dof_ids(dof)[dof.is_unknown]
	resize!(dof.unknown_indices, length(new_unknown_indices))
	dof.unknown_indices .= new_unknown_indices
end

function update_fields!(U::VecOrMat{Rtype}, d::DofManager, Uu::Vector{Rtype}) where Rtype
	@assert length(Uu) == sum(d.is_unknown)
  U[d.is_unknown] = Uu
end