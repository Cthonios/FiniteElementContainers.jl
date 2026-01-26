using Exodus
using FiniteElementContainers
using IterativeSolvers
using LinearAlgebra
using PartitionedArrays

struct Helper{A, B, C}
    global_to_local_elem::A
    global_to_local_node::A
    local_to_global_elem::B
    local_to_global_node::C
end

function dict_to_vec(d)
    dk, dv = collect(keys(d)), collect(values(d))
    perm = sortperm(dk)
    return dv[perm]
end

include("../../test/poisson/TestPoissonCommon.jl")
f(X, _) = 2. * π^2 * cos(π * X[1]) * cos(π * X[2])

mesh_file = Base.source_dir() * "/square.g"
output_file = Base.source_dir() * "/output.e"
num_dofs = 1
num_ranks = 4
distribute = identity
ranks = distribute(LinearIndices((num_ranks,)))

meshes = map(ranks) do rank
    UnstructuredMesh(mesh_file, num_ranks, rank)
end

helpers = map(meshes, ranks) do mesh, rank
    mesh.nodal_coords
    local_conns = mesh.element_conns
    elem_map = mesh.element_id_maps
    node_map = mesh.node_id_map

    global_to_local_elem = Dict{Int, Int}()
    for block_elem_map in values(elem_map)
        for e in eachindex(block_elem_map)
            if haskey(global_to_local_elem, block_elem_map[e])
                @assert false
            end
            global_to_local_elem[block_elem_map[e]] = e
        end
    end

    global_to_local_node = Dict{Int, Int}()
    for n in eachindex(node_map)
        if haskey(global_to_local_node, node_map[n])
            @assert false
        end
        global_to_local_node[node_map[n]] = n
    end

    local_to_global_elem = deepcopy(elem_map)
    local_to_global_node = copy(node_map)

    global_conns = deepcopy(local_conns)
    for block_conn in values(global_conns)
        for e in axes(block_conn, 2)
            for n in axes(block_conn, 1)
                block_conn[n, e] = local_to_global_node[block_conn[n, e]]
            end
        end
    end

    # elements owners (the rank owns all the elements on this rank and nothing else)
    n_elems = mapreduce(length, sum, values(local_to_global_elem))
    elem_owners = fill(rank, n_elems)

    # node to elem 
    node_to_elems = Dict{Int, Vector{Int}}()
    for block_conn in values(global_conns)
        for (e, nodes) in enumerate(block_conn)
            for n in nodes
                push!(get!(node_to_elems, n, Int[]), e)
            end
        end
    end
    
    return Helper(
        global_to_local_elem, global_to_local_node,
        dict_to_vec(local_to_global_elem), dict_to_vec(local_to_global_node)
    )
end

# @assert false

node_request = map(helpers, ranks) do helper, rank
    # discover node ownership
    global_to_local_node = helper.global_to_local_node
    global_node_ids = collect(keys(global_to_local_node))
    node_request = [(nid, rank) for nid in global_node_ids]
end

all_nodes = gather(node_request, destination = :all)

node_owners = map(all_nodes) do nodes
    node_owners = Dict{Int, Vector{Int}}()
    for list in nodes
        for (node, rank) in list
            if !haskey(node_owners, node)
                node_owners[node] = Int[]
            end

            push!(node_owners[node], rank)
        end
    end

    new_node_owners = Dict{Int, Int}()
    for (node, list) in node_owners
        new_node_owners[node] = maximum(list)
    end

    new_node_owners
end

local_dofs, n_local_owned = tuple_of_arrays(map(node_owners, ranks) do nodes, rank
    owned_nodes = filter(n -> nodes[n] == rank, keys(nodes) |> collect)

    local_dofs = Dict{Int, Int}()
    for (i, nid) in enumerate(owned_nodes)
        local_dofs[nid] = i
    end

    local_dofs, length(owned_nodes)
end)

n_all = gather(n_local_owned, destination = :all)
offset = sum(n_all[1:1])
# offsets = map(ranks) do rank
#     sum(n_all[1:rank])
# end

# out = map(local_dofs) do dofs

# end 
