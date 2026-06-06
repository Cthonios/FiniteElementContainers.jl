module PartitionedArraysExt

# TODO make Exodus an unnecessary import
using Exodus
using FiniteElementContainers
using PartitionedArrays


#######################################################################
# Helper methods for setup
#######################################################################
# NOTE dof_to_unknown returns -1 when the dof 
function _create_field_to_unknown(n_total_fields, dirichlet_dofs)
	unknown_to_field = Vector{eltype(dirichlet_dofs)}(undef, n_total_fields - length(dirichlet_dofs))
	ids = 1:n_total_fields
    n = 1
    for field_id in ids
        if !insorted(field_id, dirichlet_dofs)
            unknown_to_field[n] = field_id
            n += 1
        end
    end
    field_to_unknown = Dict([(x, y) for (x, y) in zip(unknown_to_field, 1:length(unknown_to_field))])

	for dof in dirichlet_dofs
		field_to_unknown[dof] = -1
	end

	new_field_to_unknown = Vector{Int}(undef, length(field_to_unknown))
	for field_id in 1:n_total_fields
		new_field_to_unknown[field_id] = field_to_unknown[field_id]
	end
	return new_field_to_unknown, unknown_to_field
end


function distribute_mesh(file_name::String, n_ranks::Int, ranks)
	map_main(ranks) do rank
		# TODO add a check to see if we need to actually decomp
		@info "Running decomp on $file_name with $n_ranks processors"
		decomp(file_name, n_ranks)
	end
	PartitionedArrays.barrier(ranks)
	return nothing
end

# TODO need to adjust for problems with more than one dof
# TODO should this only run on rank 1 and then distribute?
function global_colorings(file_name, n_dofs, n_ranks)
	global_elems_to_colors, global_nodes_to_colors = Exodus.collect_global_element_and_node_numberings(file_name, n_ranks)
	global_dofs = reshape(1:n_dofs * length(global_nodes_to_colors), n_dofs, length(global_nodes_to_colors))
	global_dofs_to_colors = similar(global_dofs)
	for dof in 1:n_dofs
		global_dofs_to_colors[dof, :] .= global_nodes_to_colors
	end
	global_dofs_to_colors = global_dofs_to_colors |> vec

	return global_dofs_to_colors, convert.(Int, global_elems_to_colors)
end

#######################################################################
# DofManager stuff
#######################################################################
abstract type AbstractPartition{
	I <: Integer,
	V <: AbstractVector{I},
	P <: AbstractVector{<:AbstractLocalIndices}
} end

struct ElementPartition{
	I, 
	V,
	P <: AbstractVector{OwnAndGhostIndices{V}}
} <: AbstractPartition{I, V, P}
	colors::V
	parts::P

	function ElementPartition(
		colors::V, parts::P
	) where {V <: AbstractVector{<:Integer}, P <: AbstractVector{OwnAndGhostIndices{V}}}
		new{eltype(colors), V, P}(colors, parts)
	end
end

struct FieldPartition{
	I,
	V,
	P <: AbstractVector{LocalIndices}
} <: AbstractPartition{I, V, P}
	colors::V
	parts::P

	function FieldPartition(
		colors::V, parts::P
	) where {V <: AbstractVector{<:Integer}, P <: AbstractVector{LocalIndices}}
		new{eltype(colors), V, P}(colors, parts)
	end
end

struct SolutionPartition{
	I,
	V,
	P <: AbstractVector{OwnAndGhostIndices{V}}
} <: AbstractPartition{I, V, P}
	colors::V
	parts::P

	function SolutionPartition(
		colors::V, parts::P
	) where {V <: AbstractVector{<:Integer}, P <: AbstractVector{OwnAndGhostIndices{V}}}
		new{eltype(colors), V, P}(colors, parts)
	end
end

struct PDofManager{
	I <: Integer,
	V <: AbstractVector{I},
	L <: AbstractVector{LocalIndices},
	O <: AbstractVector{OwnAndGhostIndices{V}},
	D,
	R,
	Var,
	DDofs
}
	elem_partition::ElementPartition{I, V, O}
	field_partition::FieldPartition{I, V, L}
	solution_partition::SolutionPartition{I, V, O}
	field_to_solution::V
	solution_to_field::V
	local_dof_managers::D
	ranks::R
	var::Var
	dirichlet_dofs::DDofs
end

# bad piracy at the moment...
# eventually we should have DofManager capture all
# this behavior
function FiniteElementContainers.DofManager(
	u::FiniteElementContainers.AbstractFunction,
	dbcs::Vector{DirichletBC},
	num_ranks, ranks, mesh_file, mesh
)
	num_dofs = num_fields(u)
	var_names = names(u)

	# get colorings
	fields_to_colors, elems_to_colors = global_colorings(mesh_file, num_dofs, num_ranks)

	# create element partition (easy)
	elem_parts = partition_from_color(ranks, elems_to_colors)
	elem_parts = ElementPartition(elems_to_colors, elem_parts)

	# create field partition (slightly more difficult)
	num_nodes_global = length(fields_to_colors) ÷ num_dofs
	ids = reshape(1:length(fields_to_colors), num_dofs, num_nodes_global)

	field_parts = map(mesh, ranks) do mesh_local, rank
		field_id_map = ids[:, mesh_local.node_id_map] |> vec
		field_to_owner = map(x -> fields_to_colors[x], field_id_map)
		LocalIndices(length(fields_to_colors), rank, field_id_map, field_to_owner)
	end
	field_parts = FieldPartition(fields_to_colors, field_parts)

	# create solution partition

	# below is attempt to use communication to get all dirichlet dofs so
	# we don't need to read in teh whole mesh
	# need to collect dirichlet dofs in global numbering
	# dirichlet_dofs = map(field_parts, u.fspace, mesh) do field_part, fspace, mesh_local
	# 	u_local = eval(typeof(u).name.name){typeof(fspace)}(fspace, var_names)
	# 	dof_local = DofManager(u_local)
	# 	dbcs_local = DirichletBCs(mesh_local, dof_local, dbcs)
	# 	dirichlet_dofs_local = FiniteElementContainers.dirichlet_dofs(dbcs_local)
	# 	ltg = local_to_global(field_part)
	# 	dirichlet_dofs_global = ltg[dirichlet_dofs_local]
	# 	return dirichlet_dofs_global
	# end

	# collect local dof managers and set them up with appropriate local
	# dirichlet bcs
	local_dof_managers = map(u.fspace, mesh) do fspace, mesh_local
		u_local = eval(typeof(u).name.name){typeof(fspace)}(fspace, var_names)
		dof_local = DofManager(u_local; use_condensed = true)
		dbcs_local = DirichletBCs(mesh_local, dof_local, dbcs)
		dirichlet_dofs_local = FiniteElementContainers.dirichlet_dofs(dbcs_local)
		update_dofs!(dof_local, dirichlet_dofs_local)
		return dof_local
	end

	# # now need to scatter all to one proc
	# # out = map(dirichlet_dofs) do ddofs
	# # 	gather(ddofs, destination = 1)
	# # end
	# out = gather(dirichlet_dofs, destination = 1)
	# dirichlet_dofs = map_main(out) do out_local
	# 	return reduce(vcat, out_local) |> unique |> sort
	# end

	# slow approach for now
	serial_mesh = UnstructuredMesh(mesh_file)
	serial_V = FunctionSpace(serial_mesh, H1Field, Lagrange)
	serial_u = eval(typeof(u).name.name){typeof(serial_V)}(serial_V, var_names)
	serial_dof = DofManager(serial_u)
	serial_dbcs = DirichletBCs(serial_mesh, serial_dof, dbcs)
	dirichlet_dofs = FiniteElementContainers.dirichlet_dofs(serial_dbcs)

	# TODO need to cleanup up and finish everthing below
	unknowns_to_colors = copy(fields_to_colors)
	deleteat!(unknowns_to_colors, dirichlet_dofs)

	solution_parts = partition_from_color(ranks, unknowns_to_colors)
	solution_parts = SolutionPartition(unknowns_to_colors, solution_parts)

	# finally create the maps from field to solution and back
	field_to_unknown, unknown_to_field = _create_field_to_unknown(length(fields_to_colors), dirichlet_dofs)
	return PDofManager(
		elem_parts, field_parts, solution_parts,
		field_to_unknown, unknown_to_field,
		local_dof_managers, ranks, u,
		dirichlet_dofs
	)
end

function FiniteElementContainers.create_field(dof::PDofManager)
	local_fields = map(dof.local_dof_managers) do dof_local
		return create_field(dof_local)
	end
	return PVector(local_fields, dof.field_partition.parts)
end

function FiniteElementContainers.create_unknowns(dof::PDofManager)
	return pzeros(dof.solution_partition.parts)
end

function FiniteElementContainers.function_space(dof::PDofManager)
	return dof.var.fspace
end

function FiniteElementContainers.update_field_unknowns!(U::PVector, dof::PDofManager, Uu::PVector)
	map(
		partition(U), dof.field_partition.parts,
		partition(Uu), dof.solution_partition.parts,
		dof.ranks
	) do U_local, field_part, Uu_local, solution_part, rank
		field_gids   = local_to_global(field_part)
		owners       = local_to_owner(field_part)
		solution_gtl = global_to_local(solution_part)

		# putting dof in fec_foreach might not be gpu safe...
		FiniteElementContainers.fec_foreach(field_gids) do i
			if owners[i] == rank
				field_id = field_gids[i]
				solution_id = dof.field_to_solution[field_id]
				if solution_id > 0
					U_local[i] = Uu_local[solution_gtl[solution_id]]
				end
			end
		end
	end

    return nothing
end

# remove me once PDofManager is up to snuff
function FiniteElementContainers.update_field_unknowns!(U::PVector, parts, Uu::PVector, ranks)
    map(
        partition(U),
        U.index_partition,
        partition(Uu),
        Uu.index_partition,
        ranks
    ) do U_local, field_part, Uu_local, unknown_part, my_rank

        field_gids = local_to_global(field_part)
        owners = local_to_owner(field_part)
        unknown_gtl = global_to_local(unknown_part)

		FiniteElementContainers.fec_foreach(field_gids) do i
            if owners[i] == my_rank
				field_id = field_gids[i]
				unknown_id = parts.field_to_unknown[field_id]
				if unknown_id > 0
					U_local[i] = Uu_local[unknown_gtl[unknown_id]]
				end
			end
        end
    end

    return nothing
end

#######################################################################
# FunctionSpace stuff
#######################################################################
function FiniteElementContainers.FunctionSpace(meshes::AbstractVector, field_type, interp_type; kwargs...)
	map(meshes) do mesh
		return FunctionSpace(mesh, field_type, interp_type; kwargs...)
	end
end

#######################################################################
# Mesh stuff
#######################################################################
function FiniteElementContainers.UnstructuredMesh(
	file_name::String, num_ranks, rank::I
) where I <: Integer
	file_name = file_name * ".$num_ranks" * ".$(lpad(rank - 1, Exodus.exodus_pad(num_ranks |> Int32), '0'))"
	return UnstructuredMesh(file_name)
end

function FiniteElementContainers.UnstructuredMesh(
    mesh_file::String, num_ranks::Int, ranks::AbstractArray{<:Integer, 1}
)
	distribute_mesh(mesh_file, num_ranks, ranks)
	return map(ranks) do rank
		UnstructuredMesh(mesh_file, num_ranks, rank)
	end
end

function FiniteElementContainers.nodal_coordinates(dof::PDofManager)
	X_vals = map(dof.var.fspace) do fspace
		fspace.coords
	end
	return PVector(X_vals, dof.field_partition.parts)
end

#######################################################################
# Assembler stuff
#######################################################################
struct PSparseMatrixPattern{
	D,
	T,
	I1 <: AbstractVector{T},
	I2 <: AbstractVector
}
	dof::D
	Is::I1
	Js::I1
	block_start_indices::I2
	block_el_level_sizes::I2
	unknown_dofs::I1
end

function PSparseMatrixPattern(dof::PDofManager)
	(; field_partition, field_to_solution) = dof

	Is, Js, block_start_indices, block_el_level_sizes, unknown_dofs =
		tuple_of_arrays(map(field_partition.parts, dof.var.fspace, dof.local_dof_managers) do field_part, fspace, local_dof
		# setup
		num_blocks = FiniteElementContainers.num_blocks(fspace)
		Is = Int[]
		Js = Int[]
		unknown_dofs = Int[]
		block_start_indices = Vector{Int}(undef, num_blocks)
		block_el_level_sizes = Vector{Int}(undef, num_blocks)

		# loop over blocks/elements
		n = 0
		field_ltg = local_to_global(field_part)
		dirichlet_dofs = field_ltg[local_dof.dirichlet_dofs]
		dirichlet_dofs = sort(dirichlet_dofs)
		for b in 1:num_blocks
			block_local_conns = connectivity(fspace.elem_conns, b)
			block_global_conns = field_ltg[block_local_conns]
			block_start_indices[b] = n + 1
			block_el_level_sizes[b] = size(block_local_conns, 1) * size(block_local_conns, 1)
			for e in axes(block_global_conns, 2)
				for i in axes(block_global_conns, 1)
					for j in axes(block_global_conns, 1)
						n += 1
						gi = block_global_conns[i, e]
						gj = block_global_conns[j, e]

						if !insorted(gi, dirichlet_dofs) &&
						   !insorted(gj, dirichlet_dofs)

							push!(Is, field_to_solution[gi])
							push!(Js, field_to_solution[gj])
							push!(unknown_dofs, n)
						end
					end
				end
			end
		end
		return Is, Js, block_start_indices, block_el_level_sizes, unknown_dofs
	end)
	return PSparseMatrixPattern(dof, Is, Js, block_start_indices, block_el_level_sizes, unknown_dofs)
end

function FiniteElementContainers.create_matrix_sparsity_pattern(dof::PDofManager)
	return PSparseMatrixPattern(dof)
end

function PartitionedArrays.psparse(pattern::PSparseMatrixPattern, vals)
	# parts = pattern.parts.unknown_parts
	parts = pattern.dof.solution_partition.parts
	vals = map(pattern.unknown_dofs, vals) do dofs, val
		val[dofs]
	end
	return psparse(pattern.Is, pattern.Js, vals, parts, parts)
end

struct PSparseVectorPattern{
	D,
	T,
	I1 <: AbstractVector{T},
	I2 <: AbstractVector
}
	dof::D
	Is::I1
	block_start_indices::I2
	block_el_level_sizes::I2
	unknown_dofs::I1
end

function PSparseVectorPattern(dof::PDofManager)
	(; field_partition, field_to_solution) = dof

	Is, block_start_indices, block_el_level_sizes, unknown_dofs = 
		tuple_of_arrays(map(field_partition.parts, dof.var.fspace, dof.local_dof_managers) do field_part, fspace, local_dof
		# setup
		num_blocks = FiniteElementContainers.num_blocks(fspace)
		Is = Int[]
		unknown_dofs = Int[]
		block_start_indices = Vector{Int}(undef, num_blocks)
		block_el_level_sizes = Vector{Int}(undef, num_blocks)

		# loop over blocks/elements
		n = 0
		field_ltg = local_to_global(field_part)
		dirichlet_dofs = field_ltg[local_dof.dirichlet_dofs]
		dirichlet_dofs = sort(dirichlet_dofs)
		for b in 1:num_blocks
			block_local_conns = connectivity(fspace.elem_conns, b)
			block_global_conns = field_ltg[block_local_conns]
			block_start_indices[b] = n + 1
			block_el_level_sizes[b] = size(block_local_conns, 1)
			for e in axes(block_global_conns, 2)
				for i in axes(block_global_conns, 1)
					n += 1
					gi = block_global_conns[i, e]
					if !insorted(gi, dirichlet_dofs)
						push!(Is, field_to_solution[gi])
						push!(unknown_dofs, n)
					end
				end
			end
		end
		return Is, block_start_indices, block_el_level_sizes, unknown_dofs
	end)

	return PSparseVectorPattern(
		dof,
		Is, block_start_indices,
		block_el_level_sizes, unknown_dofs
	)
end


function FiniteElementContainers.create_vector_sparsity_pattern(dof::PDofManager)
	return PSparseVectorPattern(dof)
end

function FiniteElementContainers.create_unknowns(pattern::PSparseVectorPattern)
	return create_unknowns(pattern.dof)
end

function PartitionedArrays.pvector(pattern::PSparseVectorPattern, vals)
	# parts = pattern.parts.unknown_parts
	# solution_parts = pattern.dof.solution_partition.parts
	vals = map(pattern.unknown_dofs, vals) do dofs, val
		val[dofs]
	end
	return pvector(pattern.Is, vals, pattern.dof.solution_partition.parts)
end

function PartitionedArrays.pzeros(pattern::PSparseVectorPattern)
	return pzeros(pattern.dof.solution_partition.parts)
end

struct PSparseMatrixAssembler{
	Assemblers,
	MatPattern <: PSparseMatrixPattern,
	VecPattern <: PSparseVectorPattern
}
	local_assemblers::Assemblers
	matrix_pattern::MatPattern
	vector_pattern::VecPattern
end

function FiniteElementContainers.SparseMatrixAssembler(dof::PDofManager)
	local_assemblers = map(dof.local_dof_managers) do local_dof
		SparseMatrixAssembler(local_dof; use_sparse_vector = true)
	end
	matrix_pattern = PSparseMatrixPattern(dof)
	vector_pattern = PSparseVectorPattern(dof)
	return PSparseMatrixAssembler(
		local_assemblers,
		matrix_pattern, vector_pattern
	)
end

function FiniteElementContainers.assemble_stiffness!(asm::PSparseMatrixAssembler, func, u, p)
	map(asm.local_assemblers, partition(u), p.local_parameters) do local_asm, local_u, local_p
		assemble_stiffness!(local_asm, func, local_u, local_p)
	end
end

function FiniteElementContainers.assemble_vector!(asm::PSparseMatrixAssembler, func, u, p)
	map(asm.local_assemblers, partition(u), p.local_parameters) do local_asm, local_u, local_p
		assemble_vector!(local_asm, func, local_u, local_p)
	end
end

function FiniteElementContainers.create_field(asm::PSparseMatrixAssembler)
	return create_field(asm.vector_pattern.dof)
end

function FiniteElementContainers.create_unknowns(asm::PSparseMatrixAssembler)
	return create_unknowns(asm.vector_pattern.dof)
end

function FiniteElementContainers.residual(asm::PSparseMatrixAssembler)
	vals = map(asm.local_assemblers) do local_asm
		local_asm.residual_unknowns
	end
	return pvector(asm.vector_pattern, vals) |> fetch
end

function FiniteElementContainers.stiffness(asm::PSparseMatrixAssembler)
	vals = map(asm.local_assemblers) do local_asm
		local_asm.stiffness_storage
	end
	return psparse(asm.matrix_pattern, vals) |> fetch
end

#######################################################################
# Parameters
#######################################################################
struct PParameters{P}
	local_parameters::P
end

function FiniteElementContainers.create_parameters(
	mesh::AbstractVector{<:FiniteElementContainers.AbstractMesh},
	asm::PSparseMatrixAssembler,
	physics,
	props;
	kwargs...
)
	# map over assemblers and what not 
	local_parameters = map(mesh, asm.local_assemblers) do local_mesh, local_asm
		create_parameters(local_mesh, local_asm, physics, props; kwargs...)
	end
	return PParameters(local_parameters)
end

end # module
