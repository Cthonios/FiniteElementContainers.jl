module PartitionedArraysExt

# TODO make Exodus an unnecessary import
using Exodus
using FiniteElementContainers
using PartitionedArrays

# want to create LocalIndices here
# TODO this only works for H1 right now
function _create_field_partition(
	file_name::String, n_dofs::Int, n_ranks::Int, ranks,
	field_to_colors
)
	n_nodes_global = length(field_to_colors) ÷ n_dofs
	ids = reshape(1:length(field_to_colors), n_dofs, n_nodes_global)

	node_id_maps = map(ranks) do rank
		mesh = UnstructuredMesh(file_name, n_ranks, rank)
		mesh.node_id_map
	end

	map(node_id_maps, ranks) do node_id_map, rank
		field_id_map = ids[:, node_id_map] |> vec
		field_to_owner = map(x -> field_to_colors[x], field_id_map)
		LocalIndices(length(field_to_colors), rank, field_id_map, field_to_owner)
	end
end

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

struct FEMPartition{
	I <: Integer,
	V <: AbstractVector{I},
	E <: AbstractVector{OwnAndGhostIndices{V}},
	F <: AbstractVector{LocalIndices},
	U <: AbstractVector{OwnAndGhostIndices{V}}
}
	dirichlet_dofs::V
	elem_colors::V
	elem_parts::E
	field_colors::V
	field_parts::F
	unknown_colors::V
	unknown_parts::U
	field_to_unknown::V
	unknown_to_field::V
end

# TODO currently this only supports H1 spaces
function FiniteElementContainers.create_partition(
	file_name::String, 
	n_dofs::Int, n_ranks::Int, ranks,
	dirichlet_dofs::AbstractVector{<:Integer}
)
	# get colorings
	fields_to_colors, elems_to_colors = global_colorings(file_name, n_dofs, n_ranks)
	unknowns_to_colors = copy(fields_to_colors)
	deleteat!(unknowns_to_colors, dirichlet_dofs)

	# create element and field partition
	elem_parts = partition_from_color(ranks, elems_to_colors)
	field_parts = _create_field_partition(file_name, n_dofs, n_ranks, ranks, fields_to_colors)

	unknown_parts = partition_from_color(ranks, unknowns_to_colors)

	field_to_unknown, unknown_to_field = _create_field_to_unknown(length(fields_to_colors), dirichlet_dofs)

	return FEMPartition(
		dirichlet_dofs,
		elems_to_colors, elem_parts, 
		fields_to_colors, field_parts, 
		unknowns_to_colors, unknown_parts,
		field_to_unknown, unknown_to_field
	)
end

function FiniteElementContainers.distribute_mesh(file_name::String, n_ranks::Int, ranks::LinearIndices)
	return _distribute_mesh(file_name, n_ranks, ranks)
end

function FiniteElementContainers.distribute_mesh(file_name::String, n_ranks::Int, ranks::MPIArray)
	_distribute_mesh(file_name, n_ranks, ranks)
	# currently need a barrier or else we have a race condition
	# MPI.Barrier(MPI.COMM_WORLD)
	return nothing
end

function _distribute_mesh(file_name::String, n_ranks::Int, ranks::AbstractArray{<:Integer, 1})
	map_main(ranks) do rank
		@info "Running decomp on $file_name with $n_ranks processors"
		decomp(file_name, n_ranks)
	end
	PartitionedArrays.barrier(ranks)
	return nothing
end

# TODO need to adjust for problems with more than one dof
# TODO should this only run on rank 1 and then distribute?
function FiniteElementContainers.global_colorings(file_name, n_dofs, n_ranks)
	global_elems_to_colors, global_nodes_to_colors = Exodus.collect_global_element_and_node_numberings(file_name, n_ranks)
	global_dofs = reshape(1:n_dofs * length(global_nodes_to_colors), n_dofs, length(global_nodes_to_colors))
	global_dofs_to_colors = similar(global_dofs)
	for dof in 1:n_dofs
		global_dofs_to_colors[dof, :] .= global_nodes_to_colors
	end
	global_dofs_to_colors = global_dofs_to_colors |> vec

	return global_dofs_to_colors, convert.(Int, global_elems_to_colors)
end

# some new constructors for other types

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

struct PSparseMatrixPattern{
	T,
	I1 <: AbstractVector{T},
	I2 <: AbstractVector,
	P
}
	Is::I1
	Js::I1
	block_start_indices::I2
	block_el_level_sizes::I2
	unknown_dofs::I1
	parts::P
end

function PSparseMatrixPattern(meshes, parts::FEMPartition)

    (; dirichlet_dofs, elem_parts, field_to_unknown) = parts

    IIs, JJs, block_start_indices, block_el_level_sizes, unknown_dofs =
        tuple_of_arrays(map(meshes, elem_parts) do mesh, part

            IIs  = Int[]
            JJs  = Int[]
            unknown_dofs = Int[]

            block_start_indices = Vector{Int}(undef, length(mesh.element_conns))
            block_el_level_sizes = Vector{Int}(undef, length(mesh.element_conns))

            n = 0
            for (key, conn) in mesh.element_conns
                for e in axes(conn, 2)

                    glob_conn = @views mesh.node_id_map[conn[:, e]]

                    for i in axes(glob_conn, 1)
                        for j in axes(glob_conn, 1)
                            n += 1

                            gi = glob_conn[i]
                            gj = glob_conn[j]

                            # only record if both are unknown
                            if !insorted(gi, dirichlet_dofs) &&
                               !insorted(gj, dirichlet_dofs)

                                push!(IIs, field_to_unknown[gi])
                                push!(JJs, field_to_unknown[gj])
                                push!(unknown_dofs, n)
                            end
                        end
                    end
                end
            end

            IIs, JJs, block_start_indices, block_el_level_sizes, unknown_dofs
        end)

    return PSparseMatrixPattern{
        eltype(IIs),
        typeof(IIs),
        typeof(block_start_indices),
        typeof(parts)
    }(
        IIs, JJs, block_start_indices,
        block_el_level_sizes, unknown_dofs, parts
    )
end

function FiniteElementContainers.create_matrix_sparsity_pattern(meshes, parts::FEMPartition)
	return PSparseMatrixPattern(meshes, parts)
end

function PartitionedArrays.psparse(pattern::PSparseMatrixPattern, vals)
	parts = pattern.parts.unknown_parts
	vals = map(pattern.unknown_dofs, vals) do dofs, val
		val[dofs]
	end
	return psparse(pattern.Is, pattern.Js, vals, parts, parts)
end

struct PSparseVectorPattern{
	T,
	I1 <: AbstractVector{T},
	I2 <: AbstractVector,
	P
}
	Is::I1
	block_start_indices::I2
	block_el_level_sizes::I2
	unknown_dofs::I1
	parts::P
end

function PSparseVectorPattern(meshes, parts::FEMPartition)
    (; dirichlet_dofs, elem_parts, field_to_unknown) = parts
	Is, block_start_indices, block_el_level_sizes, unknown_dofs = 
		tuple_of_arrays(map(meshes, elem_parts) do mesh, part
			Is = Int[]
			unknown_dofs = Int[]
			block_start_indices = Vector{Int}(undef, length(mesh.element_conns))
            block_el_level_sizes = Vector{Int}(undef, length(mesh.element_conns))

			n = 0   # IMPORTANT: start at 0

			for (key, conn) in mesh.element_conns
				for e in axes(conn, 2)
					glob_conn = @views mesh.node_id_map[conn[:, e]]
					for i in axes(glob_conn, 1)
						n += 1
						gi = glob_conn[i]
						if !insorted(gi, dirichlet_dofs)
							push!(Is, field_to_unknown[gi])
							push!(unknown_dofs, n)
						end
					end
				end
			end
			return Is, block_start_indices, block_el_level_sizes, unknown_dofs
		end)
	return PSparseVectorPattern{
		eltype(Is),
		typeof(Is),
		typeof(block_start_indices),
		typeof(parts)
	}(
		Is, block_start_indices,
		block_el_level_sizes, unknown_dofs, parts
	)
end

# function PSparseVectorPattern(meshes, parts::FEMPartition)

#     (; dirichlet_dofs, elem_parts, field_to_unknown) = parts

#     # Precompute fast Dirichlet lookup
#     ndofs = maximum(keys(field_to_unknown))
#     is_dirichlet = falses(ndofs)
#     is_dirichlet[dirichlet_dofs] .= true

#     Is, block_start_indices, block_el_level_sizes, unknown_dofs =
#         tuple_of_arrays(map(meshes, elem_parts) do mesh, part

#             Is = Int[]
#             unknown_dofs = Int[]

#             block_start_indices = Vector{Int}(undef, length(mesh.element_conns))
#             block_el_level_sizes = Vector{Int}(undef, length(mesh.element_conns))

#             n = 0   # linear index into full element vector ordering

#             for (key, conn) in mesh.element_conns
#                 for e in axes(conn, 2)

#                     glob_conn = @views mesh.node_id_map[conn[:, e]]

#                     for i in axes(glob_conn, 1)

#                         # must match element vector build ordering
#                         n += 1

#                         gi = glob_conn[i]

#                         if !is_dirichlet[gi]
#                             push!(Is, field_to_unknown[gi])
#                             push!(unknown_dofs, n)
#                         end
#                     end
#                 end
#             end

#             Is, block_start_indices, block_el_level_sizes, unknown_dofs
#         end)

#     return PSparseVectorPattern{
#         eltype(Is),
#         typeof(Is),
#         typeof(block_start_indices),
#         typeof(parts)
#     }(
#         Is,
#         block_start_indices,
#         block_el_level_sizes,
#         unknown_dofs,
#         parts
#     )
# end

function FiniteElementContainers.create_vector_sparsity_pattern(meshes, parts::FEMPartition)
	return PSparseVectorPattern(meshes, parts)
end

function PartitionedArrays.pvector(pattern::PSparseVectorPattern, vals)
	parts = pattern.parts.unknown_parts
	vals = map(pattern.unknown_dofs, vals) do dofs, val
		val[dofs]
	end
	return pvector(pattern.Is, vals, parts)
end

function PartitionedArrays.pzeros(pattern::PSparseVectorPattern)
	return pzeros(pattern.parts.unknown_parts)
end

function FiniteElementContainers.create_unknowns(parts::FEMPartition)
	return pzeros(parts.unknown_parts)
end

# dof manager like stuff
# function FiniteElementContainers.extract_field_unknowns!(
# 	UU,
# 	parts::FEMPartition,
# 	U
# )

# end

# some mesh helpers
function FiniteElementContainers.nodal_coordinates(meshes, parts)
	X_vals = map(meshes) do mesh
		mesh.nodal_coords
	end
	return PVector(X_vals, parts.field_parts)
end

end # module
