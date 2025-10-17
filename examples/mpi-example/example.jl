import FiniteElementContainers:
    communication_graph,
    cpu, 
    decompose_mesh,
    distribute,
    global_colorings,
    DistributedDevice,
    getdata,
    num_dofs_per_rank,
    ParVector,
    rank_devices,
    scatter_ghosts!
using Exodus
using KernelAbstractions
using MPI

backend = CPU()
num_dofs = 1
num_ranks = 4
ranks = 1:4 |> collect
mesh_file = Base.source_dir() * "/square.g"

decompose_mesh(mesh_file, num_ranks)
global_dofs_to_colors = global_colorings(mesh_file, num_dofs, num_ranks)

comm = MPI.COMM_WORLD
# comm = nothing

ranks = distribute(ranks, comm)
# n_dofs_per_rank = distribute(num_dofs_per_rank(global_dofs_to_colors), comm)

# shards = shard_indices(global_dofs_to_colors, )
comm_graphs = map(ranks) do rank
    mesh_file = Base.source_dir() * "/square.g"
    mesh_file = mesh_file * ".$num_ranks" * ".$(lpad(rank - 1, Exodus.exodus_pad(num_ranks |> Int32), '0'))"
    comm_graph = communication_graph(mesh_file, global_dofs_to_colors, num_dofs, rank)
end

par_vec = ParVector(comm_graphs)
parts = map(par_vec.parts, ranks) do part, rank
    part .= rank
    part
end
par_vec = ParVector(parts)
par_vec_2 = scatter_ghosts!(par_vec)
MPI.Barrier(comm)
# par_vec_2 = par_vec_2.parts
print("part on $(MPI.Comm_rank(comm) + 1) is $par_vec_2\n")
