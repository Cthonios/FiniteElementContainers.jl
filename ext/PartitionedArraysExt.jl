module PartitionedArraysExt

using Exodus
using FiniteElementContainers
using PartitionedArrays

function FiniteElementContainers.decompose_mesh(file_name::String, n_ranks::Int, comm)
    if comm == 1
        @info "Running decomp on $file_name with $n_ranks processors"
        decomp(file_name, n_ranks)
    end
end
  
# TODO need to adjust for problems with more than one dof
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

# function FiniteElementContainers.global_colorings(file_name::String, num_dofs::Int, num_procs::Int, comm)
#     # if comm == 1
#         @info "Setting up global colorings on rank $comm"
#         global_elems_to_colors, global_nodes_to_colors = Exodus.collect_global_element_and_node_numberings(file_name, num_procs)
#         global_dofs = reshape(1:num_dofs * length(global_nodes_to_colors), num_dofs, length(global_nodes_to_colors))
#         global_dofs_to_colors = similar(global_dofs)

#         for dof in 1:num_dofs
#             global_dofs_to_colors[dof, :] .= global_nodes_to_colors
#         end
#         global_dofs_to_colors = global_dofs_to_colors |> vec
#     # else
#     #     exo = ExodusDatabase(file_name, "r")
#     #     num_elems = Exodus.initialization(exo).num_elements
#     #     num_nodes = Exodus.initialization(exo).num_nodes
#     #     n_dofs = num_dofs * num_nodes
#     #     global_elems_to_colors = Vector{Int}(undef, num_elems)
#     #     global_dofs_to_colors = Vector{Int}(undef, n_dofs)
#     #     close(exo)
#     # end

#     # MPI.Bcast!(global_dofs_to_colors, root, comm)
#     # return global_dofs_to_colors
#     # gather(global_dofs_to_colors, destination=:all)
# end

function FiniteElementContainers.UnstructuredMesh(
    file_name, num_ranks, rank
)
    file_name = file_name * ".$num_ranks" * ".$(lpad(rank - 1, Exodus.exodus_pad(num_ranks |> Int32), '0'))"
    return UnstructuredMesh(file_name)
end

struct ExodusPartition{A, B, C, D, E, F, G}
	# these map globals to globals
	exo_dof_to_own::A
	exo_elem_to_own::A
	exo_elem_to_exo_node::B
	exo_node_to_exo_elem::C
	# tracks total dofs/elems per part
	ndpp::A
	nepp::A
	# different partitions we have to keep track of
	dof_parts::D
	elem_parts::E
	# maps between PartitionedArrays and exodus
	dof_exo_to_par::F
	dof_par_to_exo::F
	elem_exo_to_par::F
	elem_par_to_exo::F
	# for building sparse parrays
	par_conns::G
end

function Base.show(io::IO, part::ExodusPartition)
	println(io, "ExodusPartition:")
	println(io, "  Number of partitions  = $(maximum(values(part.exo_to_par)))")
	# println(io, "  Number of global dofs = $(length(part.exo_to_par))")
end

function FiniteElementContainers.create_partition(
	mesh_file, num_dofs, num_ranks, ranks;
	add_element_borders = false,
	add_ghost_nodes = false
)
	exo_dof_to_own, exo_elem_to_own = FiniteElementContainers.global_colorings(mesh_file, num_dofs, num_ranks)
	exo_elem_to_glob_node = FiniteElementContainers._global_elem_to_global_node(mesh_file)
	exo_node_to_glob_elem = FiniteElementContainers._global_node_to_global_elem(mesh_file)

	ndpp, nepp = tuple_of_arrays(map(ranks) do rank
		nd = count(x -> x == rank, exo_dof_to_own)
		ne = count(x -> x == rank, exo_elem_to_own)
		return nd, ne 
	end)

	dof_parts = variable_partition(ndpp, sum(ndpp))
	elem_parts = variable_partition(nepp, sum(nepp))

	meshes = map(ranks) do rank
		UnstructuredMesh(mesh_file, num_ranks, rank)
	end

	dof_exo_to_par, dof_par_to_exo = FiniteElementContainers._dofs_exo_to_par_dicts(
		meshes, num_dofs,
		exo_dof_to_own,
		ndpp, dof_parts, ranks
	)
	elem_exo_to_par, elem_par_to_exo = FiniteElementContainers._elems_exo_to_par_dicts(
		meshes,
		exo_elem_to_own,
		nepp, elem_parts, ranks
	)

	par_conns = map(meshes) do mesh
		conns = mesh.element_conns
		node_map = mesh.node_id_map
		# update in place since we'll make new mesh objects later
		for conn in values(conns)
			for e in axes(conn, 2)
				for n in axes(conn, 1)
					conn[n, e] = dof_exo_to_par[node_map[conn[n, e]]]
				end
			end
		end
		conns
	end

	# get inernal global exo nodes
	

	exo_parts = ExodusPartition(
		exo_dof_to_own, exo_elem_to_own,
		exo_elem_to_glob_node,
		exo_node_to_glob_elem,
		ndpp, nepp,
		dof_parts, elem_parts,
		dof_exo_to_par, dof_par_to_exo,
		elem_exo_to_par, elem_par_to_exo,
		par_conns
	)

	if add_element_borders
		exo_parts = _add_element_borders(exo_parts, meshes, ranks)
	end

	if add_ghost_nodes
		exo_parts = _add_ghost_nodes(exo_parts, meshes, ranks)
	end

	return exo_parts
end

function _add_element_borders(exo_parts, meshes, ranks)
	owns, ghosts = tuple_of_arrays(map(meshes, ranks) do mesh, rank
		conns = mesh.element_conns
		node_map = mesh.node_id_map
		elem_map = Exodus.read_id_map(mesh.mesh_obj.mesh_obj, ElementMap)
		owns = Vector{Int}(undef, 0)
		ghosts = Vector{Int}(undef, 0)
	
		for node in node_map
			elems = exo_parts.exo_node_to_exo_elem[node]
			for elem in elems
				if exo_parts.exo_elem_to_own[elem] == rank
					push!(owns, elem)
				else
					push!(ghosts, elem)
				end
			end
		end
	
		owns = unique(sort(owns))
		ghosts = unique(sort(ghosts))
		owns = map(x -> exo_parts.elem_exo_to_par[x], owns)
		ghosts = map(x -> exo_parts.elem_exo_to_par[x], ghosts)
	
		return owns, ghosts
	end)

	elem_parts = map(exo_parts.elem_parts, ghosts) do part, ghost
		owners = map(x -> exo_parts.exo_elem_to_own[exo_parts.elem_par_to_exo[x]], ghost)
		# perm = sortperm(owners)
		# union_ghost(part, ghost[perm], owners[perm])
		union_ghost(part, ghost, owners)
	end

	return ExodusPartition(
		exo_parts.exo_dof_to_own, exo_parts.exo_elem_to_own,
		exo_parts.exo_elem_to_exo_node,
		exo_parts.exo_node_to_exo_elem,
		exo_parts.ndpp, exo_parts.nepp,
		exo_parts.dof_parts, elem_parts,
		exo_parts.dof_exo_to_par, exo_parts.dof_par_to_exo,
		exo_parts.elem_exo_to_par, exo_parts.elem_par_to_exo,
		exo_parts.par_conns
	)
end

function _add_ghost_nodes(exo_parts, meshes, ranks)
	ghost_nodes, owners = tuple_of_arrays(map(meshes, ranks) do mesh, rank
		node_map = mesh.node_id_map
		node_cmaps = Exodus.read_node_cmaps(rank, mesh.mesh_obj.mesh_obj)
		ghost_nodes, owners = Int[], Int[]
		for cmap in node_cmaps
			for (node, owner) in zip(cmap.node_ids, cmap.proc_ids)
				if exo_parts.exo_dof_to_own[node_map[node]] != rank
					push!(ghost_nodes, node_map[node])
					push!(owners, owner)
				end
			end
		end
		idx = unique(z -> ghost_nodes[z], 1:length(ghost_nodes))
		ghost_nodes = ghost_nodes[idx]
		owners = owners[idx]

		ghost_nodes = map(x -> exo_parts.dof_exo_to_par[x], ghost_nodes)

		return ghost_nodes, owners
	end)

	dof_parts = map(exo_parts.dof_parts, ghost_nodes, owners) do part, ghost, owner
		union_ghost(part, ghost, owner)
	end

	return ExodusPartition(
		exo_parts.exo_dof_to_own, exo_parts.exo_elem_to_own,
		exo_parts.exo_elem_to_exo_node,
		exo_parts.exo_node_to_exo_elem,
		exo_parts.ndpp, exo_parts.nepp,
		dof_parts, exo_parts.elem_parts,
		exo_parts.dof_exo_to_par, exo_parts.dof_par_to_exo,
		exo_parts.elem_exo_to_par, exo_parts.elem_par_to_exo,
		exo_parts.par_conns
	)
end

function FiniteElementContainers._dofs_exo_to_par_dicts(
	meshes, num_dofs,
	global_dofs_to_colors,
	n_dofs_per_parts, parts, ranks
)
	exo_to_pars, par_to_exos = tuple_of_arrays(map(meshes, parts, ranks) do mesh, part, rank
		exo_to_par = Dict{Int, Int}()
		par_to_exo = Dict{Int, Int}()
	
		par_nodes = 1:sum(n_dofs_per_parts) |> collect
		# mesh = UnstructuredMesh(mesh_file, num_ranks, rank)
		node_id_map = mesh.node_id_map

		num_nodes = length(global_dofs_to_colors) ÷ num_dofs
		global_dofs = reshape(1:length(global_dofs_to_colors), num_dofs, num_nodes)
		node_id_map = global_dofs[:, node_id_map] |> vec
	
		par_local_nodes = part.ranges[1] |> collect
		exo_local_nodes = filter(x -> global_dofs_to_colors[x] == rank, node_id_map)
	
		@assert length(exo_local_nodes) == length(par_local_nodes)
	
		for (e, p) in zip(exo_local_nodes, par_local_nodes)
			exo_to_par[e] = p
			par_to_exo[p] = e
		end
	
		return exo_to_par, par_to_exo
	end)

	whole_exo_to_par = Dict{Int, Int}()
	whole_par_to_exo = Dict{Int, Int}()

	for (exo_to_par, par_to_exo) in zip(exo_to_pars, par_to_exos)
		for (k, v) in exo_to_par
			if haskey(whole_exo_to_par, k)
				@assert false
			end
			whole_exo_to_par[k] = v
		end

		for (k, v) in par_to_exo
			if haskey(whole_par_to_exo, k)
				@assert false
			end
			whole_par_to_exo[k] = v
		end
	end

	# now check
	for (k, v) in whole_exo_to_par
		@assert whole_par_to_exo[v] == k
	end
	return whole_exo_to_par, whole_par_to_exo
end

function FiniteElementContainers._elems_exo_to_par_dicts(
	# mesh_file, num_ranks,
	meshes,
	global_elems_to_colors,
	n_elems_per_parts, parts, ranks
)
	exo_to_pars, par_to_exos = tuple_of_arrays(map(meshes, parts, ranks) do mesh, part, rank
		exo_to_par = Dict{Int, Int}()
		par_to_exo = Dict{Int, Int}()

		par_elems = 1:sum(n_elems_per_parts) |> collect
		# mesh = UnstructuredMesh(mesh_file, num_ranks, rank)
		# node_id_map = mesh.node_id_map
		elem_id_map = Exodus.read_id_map(mesh.mesh_obj.mesh_obj, ElementMap)
		par_local_elems = part.ranges[1] |> collect
		exo_local_elems = filter(x -> global_elems_to_colors[x] == rank, elem_id_map)
		
		for (e, p) in zip(exo_local_elems, par_local_elems)
			exo_to_par[e] = p
			par_to_exo[p] = e
		end
	
		return exo_to_par, par_to_exo
	end)

	whole_exo_to_par = Dict{Int, Int}()
	whole_par_to_exo = Dict{Int, Int}()

	for (exo_to_par, par_to_exo) in zip(exo_to_pars, par_to_exos)
		for (k, v) in exo_to_par
			if haskey(whole_exo_to_par, k)
				@assert false
			end
			whole_exo_to_par[k] = v
		end

		for (k, v) in par_to_exo
			if haskey(whole_par_to_exo, k)
				@assert false
			end
			whole_par_to_exo[k] = v
		end
	end

	# now check
	for (k, v) in whole_exo_to_par
		@assert whole_par_to_exo[v] == k
	end

	return whole_exo_to_par, whole_par_to_exo
end

function FiniteElementContainers._global_elem_to_global_node(mesh_file)
	exo = ExodusDatabase(mesh_file, "r")
	num_elem = Exodus.num_elements(Exodus.initialization(exo))
	elem_to_node = Dict{Int, Set{Int}}()
	for elem in 1:num_elem
		elem_to_node[elem] = Set{Int}()
	end

	blocks = read_sets(exo, Block)
	block_ids = map(x -> x.id, blocks)
	block_maps = Exodus.read_block_id_map.((exo,), block_ids)
	for (block, emap) in zip(blocks, block_maps)
		conn = block.conn
		for e in axes(conn, 2)
			for n in axes(conn, 1)
				global_e = emap[e]
				elem_to_node[global_e] = union(elem_to_node[global_e], conn[n, e])
			end
		end
	end
	close(exo)

	new_elem_to_node = Vector{Vector{Int}}(undef, num_elem)
	for elem in 1:num_elem
		new_elem_to_node[elem] = sort(collect(elem_to_node[elem]))
	end

	return new_elem_to_node
end

function FiniteElementContainers._global_node_to_global_elem(mesh_file)
	exo = ExodusDatabase(mesh_file, "r")
	num_node = Exodus.num_nodes(Exodus.initialization(exo))
	node_to_elem = Dict{Int, Set{Int}}()
	for node in 1:num_node
		node_to_elem[node] = Set{Int}()
	end

	blocks = read_sets(exo, Block)
	block_ids = map(x -> x.id, blocks)
	block_maps = Exodus.read_block_id_map.((exo,), block_ids)
	for (block, emap) in zip(blocks, block_maps)
		conn = block.conn
		for e in axes(conn, 2)
			for n in axes(conn, 1)
				global_e = emap[e]
				node_to_elem[conn[n, e]] = union(node_to_elem[conn[n, e]], global_e)
			end
		end
	end

	new_node_to_elem = Vector{Vector{Int}}(undef, num_node)
	for node in 1:num_node
		new_node_to_elem[node] = sort(collect(node_to_elem[node]))
	end

	close(exo)

	new_node_to_elem
end

function FiniteElementContainers._renumber_mesh(mesh, exo_to_par)

	for conn in mesh.element_conns
		for e in axes(conn, 2)
			for n in axes(conn, 1)
				conn[n, e] = exo_to_par[conn[n, e]]
			end
		end
	end

	for n in axes(mesh.node_id_map, 1)
		mesh.node_id_map[n] = exo_to_par[mesh.node_id_map[n]]
	end

	for nset in values(mesh.nodeset_nodes)
		for n in axes(nset, 1)
			nset[n] = exo_to_par[nset[n]]
		end
	end

	for sset in values(mesh.sideset_nodes)
		for n in axes(sset, 1)
			sset[n] = exo_to_par[sset[n]]
		end
	end

	for sset in values(mesh.sideset_side_nodes)
		for s in axes(sset, 2)
			for n in axes(sset, 1)
				sset[n, s] = exo_to_par[sset[n, s]]
			end
		end
	end

	new_mesh = UnstructuredMesh(
		mesh.mesh_obj,
		mesh.nodal_coords,
		mesh.element_block_names,
		mesh.element_types,
		mesh.element_conns, # updated in place above
		mesh.element_id_maps,
		mesh.node_id_map,   # updated in place above
		mesh.nodeset_nodes, # updated in place above
		mesh.sideset_elems, 
		mesh.sideset_nodes, # updated in place above
		mesh.sideset_sides,
		mesh.sideset_side_nodes, # updated in place above
		mesh.edge_conns, # TODO need to renumber
		mesh.face_conns, # TODO need to renumber
	)
	return new_mesh
end

end # module
