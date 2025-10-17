# TODO eventually don't specialize to MPI
# right now this is just for gathering to the element level
function communication_graph(
    file_name::String, 
    global_dofs_to_colors, 
    n_dofs, rank
)
    # open mesh and read id maps
    mesh = UnstructuredMesh(file_name)
    node_id_map = mesh.node_id_map
    cmaps = node_cmaps(mesh.mesh_obj, rank)

    # getting local dofs
    n_nodes = length(node_id_map)
    local_dofs = 1:n_dofs * n_nodes
    local_dofs = reshape(local_dofs, n_dofs, n_nodes)

    # find global dofs owned by this rank
    n_nodes_total = length(global_dofs_to_colors) ÷ n_dofs
    global_dofs = 1:length(global_dofs_to_colors)
    global_dofs = reshape(global_dofs, n_dofs, n_nodes_total)
    global_dofs_in_rank = findall(x -> x == rank, global_dofs_to_colors)

    # setup graph edges
    graph_edges = Vector{
        CommunicationGraphEdge{Vector{Float64}, Vector{Int64}}
    }(undef, length(cmaps))

    for (n, cmap) in enumerate(cmaps)
        # check to make sure proc is unique
        procs = unique(cmap.proc_ids)
        @assert length(procs) == 1
        dest = procs[1]
        local_cmap_nodes = cmap.node_ids
        global_cmap_nodes = node_id_map[cmap.node_ids]

        local_cmap_dofs = local_dofs[:, local_cmap_nodes] |> vec
        global_cmap_dofs = global_dofs[:, global_cmap_nodes] |> vec

        # figure out which part of comm maps are owned by this rank
        is_owned_send = Vector{Int}(undef, length(local_cmap_dofs))
        for (n, dof) in enumerate(global_cmap_dofs)
            if dof in global_dofs_in_rank
                is_owned_send[n] = 1
            else    
                is_owned_send[n] = 0
            end
        end

        # TODO make vector type generic, but floats are likely the main thing
        data_send = zeros(Float64, length(local_cmap_dofs))
        data_recv = similar(data_send)
        # graph_edges[n] = CommunicationGraphEdge(data_recv, data_send, local_cmap_dofs, is_owned, dest)
        is_owned_recv = similar(is_owned_send)
        
        # TODO move to interface so its not MPI specific
        comm = MPI.COMM_WORLD
        recv_req = MPI.Irecv!(is_owned_recv, comm; source=dest - 1)
        send_req = MPI.Isend(is_owned_send, comm; dest=dest - 1)
        MPI.Waitall([recv_req, send_req])

        graph_edges[n] = CommunicationGraphEdge(
            data_recv, data_send,
            local_cmap_dofs,
            is_owned_recv, is_owned_send,
            dest
        )
    end

    return CommunicationGraph(graph_edges, global_dofs_to_colors, length(node_id_map), length(global_dofs_in_rank))
end

# serial/debug case
function distribute(data)
    return data
end

# serial/debug case
function distribute(data, ::Nothing)
    return data
end

function distribute(data, comm::MPI.Comm)
    rank = MPI.Comm_rank(comm) + 1
    return MPIValue(comm, data[rank])
end

function num_dofs_per_rank(global_dofs_to_colors)
    num_ranks = length(unique(global_dofs_to_colors))
    num_dofs = Vector{Int}(undef, num_ranks)
    for n in axes(num_dofs, 1)
        num_dofs[n] = length(filter(x -> x == n, global_dofs_to_colors))
    end
    return num_dofs
end

function rank_devices(
    num_ranks, ::Val{:debug},
    backend::KA.Backend
)
    if !MPI.Initialized()
        # @assert false "You're trying to use MPI from debug mode"
        @warn "Running in debug mode. Make sure you're not using mpiexecjl"
    end
    map(x -> DistributedDevice(backend, num_ranks, x), 1:num_ranks)
end

function rank_devices(
    num_ranks, ::Val{:mpi},
    backend::KA.Backend
)
    # put init here so users don't need to be aware
    if !MPI.Initialized()
        MPI.Init()
    end

    # MPI indexing stuff
    comm = MPI.COMM_WORLD
    @assert MPI.Comm_size(comm) == num_ranks
    rank = MPI.Comm_rank(comm) + 1
    device = DistributedDevice(backend, num_ranks, rank)

    # return MPIVector(comm, device)
    return MPIValue(device)
end

function rank_devices(
    num_ranks, par_type::Symbol;
    backend = KA.CPU()
)
    val = Val{par_type}()
    return rank_devices(num_ranks, val, backend)
end

# function shard_indices(file_name, global_dofs_to_colors, rank)
#     indices = findall(x -> x == rank, global_dofs_to_colors)
#     mesh = UnstructuredMesh(file_name)
#     node_id_map = mesh.node_id_map
#     cmaps = node_cmaps(mesh.mesh_obj, rank)

#     if length(indices) == length(node_id_map)
#         # special case that this rank owns all its dofs
#     else

#     end
# end