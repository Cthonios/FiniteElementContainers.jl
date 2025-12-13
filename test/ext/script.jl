using Exodus
using FiniteElementContainers
using PartitionedArrays

include("../../test/poisson/TestPoissonCommon.jl")
# f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
f(X, _) = 1.
bc_func(_, _) = 0.

mesh_file = Base.source_dir() * "/square.g"
output_file = Base.source_dir() * "/output.e"
num_dofs = 1
num_ranks = 4
distribute = identity

ranks = distribute(LinearIndices((num_ranks,)))

# decompose mesh and global dofs to colors
# NOTE this is all happening on rank 0 currently
# put this one in seperate map as barrier
map(ranks) do rank
    FiniteElementContainers.decompose_mesh(mesh_file, num_ranks, rank)
end

# STEPS
# 1. Make global node to rank owners (plural)
# along with other stuff

global_node_owners = FiniteElementContainers.global_colorings(mesh_file, 1, num_ranks)
global_nodes_to_colors = map(minimum, global_node_owners)
global_dof_owners = FiniteElementContainers.global_colorings(mesh_file, num_dofs, num_ranks)
global_dofs_to_colors = map(minimum, global_dof_owners)

n_dofs_per_parts = map(ranks) do rank
    n_dofs_per_parts = count(x -> x == rank, global_dofs_to_colors)
    return n_dofs_per_parts
end
# 2. Make variable partition
parts = variable_partition(n_dofs_per_parts, sum(n_dofs_per_parts))
# 3. Make exo_to_par and par_to_exo maps
# for now need to make one that is one dof for a node partition
# and for a dof partition, TODO
dof_exo_to_par, dof_par_to_exo = FiniteElementContainers._exo_to_par_dicts(
    mesh_file, num_dofs, num_ranks, 
    global_dofs_to_colors,
    n_dofs_per_parts, parts, ranks
)
node_exo_to_par, node_par_to_exo = FiniteElementContainers._exo_to_par_dicts(
    mesh_file, 1, num_ranks, 
    global_nodes_to_colors,
    n_dofs_per_parts, parts, ranks
)

# 4. map meshes over to newer numbering maybe?
meshes = map(ranks) do rank
    mesh = UnstructuredMesh(mesh_file, num_ranks, rank)
    # FiniteElementContainers._renumber_mesh(mesh, node_exo_to_par)
end

# 5. Need to update ghost nodes in variable_partition
parts = map(meshes, parts, ranks) do mesh, part, rank
    node_map = mesh.node_id_map
    cmaps = Exodus.read_node_cmaps(rank, mesh.mesh_obj.mesh_obj)

    ghost_nodes = Vector{Int}(undef, 0)
    ghost_owners = Vector{Int}(undef, 0)

    for cmap in cmaps
        exo_global_nodes = node_map[cmap.node_ids]
        par_global_nodes = map(x -> dof_exo_to_par[x], exo_global_nodes)
        for (exo_node, par_node, proc) in zip(exo_global_nodes, par_global_nodes, cmap.proc_ids)
            if global_dofs_to_colors[exo_node] == proc
                push!(ghost_nodes, par_node)
                push!(ghost_owners, proc)
            end
        end
    end
    ghost_nodes
    union_ghost(part, ghost_nodes, ghost_owners)
end

v = pones(parts)

# map(meshes) do mesh
#     V = FunctionSpace(mesh, H1Field, Lagrange)
#     physics = Poisson(f)
#     props = create_properties(physics)
#     u = ScalarFunction(V, :u)
#     asm = SparseMatrixAssembler(u; use_condensed=true)
  
# end