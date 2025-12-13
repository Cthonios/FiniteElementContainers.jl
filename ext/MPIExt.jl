module MPIExt

using Exodus
using FiniteElementContainers
using MPI

# include("parallel/Parallel.jl")
# TODO eventually type this further so it only works on exodus

function FiniteElementContainers.communication_graph(file_name::String)
    comm = MPI.COMM_WORLD
    num_procs = MPI.Comm_size(comm) |> Int32
    rank = MPI.Comm_rank(comm) + 1

    file_name = file_name * ".$num_procs" * ".$(lpad(rank - 1, Exodus.exodus_pad(num_procs), '0'))"
    exo = ExodusDatabase(file_name, "r")

    node_map = read_id_map(exo, NodeMap)
    node_cmaps = Exodus.read_node_cmaps(rank, exo)

    # error checking
    for node_cmap in node_cmaps
        @assert length(unique(node_cmap.proc_ids)) == 1 "Node communication map has more than one processor present"
    end 

    comm_graph_edges = FiniteElementContainers.CommunicationGraphEdge[]
    for node_cmap in node_cmaps
        to_rank = unique(node_cmap.proc_ids)[1]
        indices = convert.(Int64, node_cmap.node_ids)
        edge = FiniteElementContainers.CommunicationGraphEdge(indices, to_rank)
        push!(comm_graph_edges, edge)
        # display(edge)
    end
    close(exo)

    return FiniteElementContainers.CommunicationGraph(comm_graph_edges)
end
  
# function FiniteElementContainers.decompose_mesh(file_name::String, n_ranks::Int)
#     if !MPI.Initialized()
#         MPI.Init()
#     end

#     comm = MPI.COMM_WORLD
#     root = 0
#     # num_procs = MPI.Comm_size(comm)
#     if MPI.Comm_rank(comm) == root
#         @info "Running decomp on $file_name with $n_ranks"
#         decomp(file_name, n_ranks)
#     end
#     MPI.Barrier(comm)
# end
  
# function FiniteElementContainers.global_colorings(file_name::String, num_dofs::Int, num_procs::Int)
#     if !MPI.Initialized()
#         MPI.Init()
#     end

#     comm = MPI.COMM_WORLD
#     root = 0
#     # num_procs = MPI.Comm_size(comm)
#     if MPI.Comm_rank(comm) == root
#         @info "Setting up global colorings on root"
#         global_elems_to_colors, global_nodes_to_colors = Exodus.collect_global_element_and_node_numberings(file_name, num_procs)
#         global_dofs = reshape(1:num_dofs * length(global_nodes_to_colors), num_dofs, length(global_nodes_to_colors))
#         global_dofs_to_colors = similar(global_dofs)

#         for dof in 1:num_dofs
#             global_dofs_to_colors[dof, :] .= global_nodes_to_colors
#         end
#         global_dofs_to_colors = global_dofs_to_colors |> vec
#     else
#         exo = ExodusDatabase(file_name, "r")
#         num_elems = Exodus.initialization(exo).num_elements
#         num_nodes = Exodus.initialization(exo).num_nodes
#         n_dofs = num_dofs * num_nodes
#         global_elems_to_colors = Vector{Int}(undef, num_elems)
#         global_dofs_to_colors = Vector{Int}(undef, n_dofs)
#         close(exo)
#     end

#     MPI.Bcast!(global_dofs_to_colors, root, comm)
#     return global_dofs_to_colors
# end

end # module
