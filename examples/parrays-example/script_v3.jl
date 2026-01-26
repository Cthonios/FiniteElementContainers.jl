using Exodus
using FiniteElementContainers
using IterativeSolvers
using Metis
using PartitionedArrays

function create_part_perm(colors, part, rank)
    owns, ghosts = Int[], Int[]
    for dof in local_to_global(part)
        if colors[dof] == rank
            push!(owns, dof)
        else
            push!(ghosts, dof)
        end
    end

    permuted_globals = vcat(owns, ghosts)
    global_to_local = Dict(g => i for (i, g) in enumerate(permuted_globals))
    perm = map(g -> global_to_local[g], part)
    return PermutedLocalIndices(part, perm)
end

include("../../test/poisson/TestPoissonCommon.jl")
f(X, _) = 2. * π^2 * cos(π * X[1]) * cos(π * X[2])

h = 0.1
Ae = (h^2/6)*[
     4.0  -1.0  -1.0  -2.0
    -1.0   4.0  -2.0  -1.0
    -1.0  -2.0   4.0  -1.0
    -2.0  -1.0  -1.0   4.0
]

mesh_file = Base.source_dir() * "/square.g"
output_file = Base.source_dir() * "/output.e"
num_dofs = 1
num_ranks = 4
distribute = identity

ranks = distribute(LinearIndices((num_ranks,)))

dofs_to_colors, elems_to_colors = tuple_of_arrays(map(ranks) do rank
    if rank == 1
        FiniteElementContainers.decompose_mesh(mesh_file, num_ranks, rank)
        dofs_to_colors, elems_to_colors = FiniteElementContainers.global_colorings(mesh_file, num_dofs, num_ranks)
    else
        nothing, nothing
    end
end)
dofs_to_colors = multicast(dofs_to_colors, source=1)
elems_to_colors = multicast(elems_to_colors, source=1)

asms, meshes, ps = tuple_of_arrays(map(ranks) do rank
    mesh = UnstructuredMesh(mesh_file, num_ranks, rank)
    V = FunctionSpace(mesh, H1Field, Lagrange)
    u = ScalarFunction(V, :u)
    asm = SparseMatrixAssembler(u)
    physics = Poisson(f)
    props = create_properties(physics)
    p = create_parameters(mesh, asm, physics, props)
    return asm, mesh, p
end)

dof_partition = map(dofs_to_colors, meshes, ranks) do colors, mesh, rank
    n_global = length(colors)
    local_to_global = mesh.node_id_map
    local_to_owner = map(x -> colors[x], local_to_global)
    LocalIndices(n_global, rank, local_to_global, local_to_owner)
end

elem_partition = map(elems_to_colors, meshes, ranks) do colors, mesh, rank
    n_global = length(colors)
    local_to_global = mesh.element_id_map
    local_to_owner = map(x -> colors[x], local_to_global)
    LocalIndices(n_global, rank, local_to_global, local_to_owner)
end

# need to map LocalIndices to PermutedLocalIndices
# since LocalIndices isn't completely supported

dof_partition = map(dofs_to_colors, dof_partition, ranks) do colors, part, rank
    create_part_perm(colors, part, rank)
end

assembly_neighbors(dof_partition; symmetric = true)
assembly_neighbors(elem_partition; symmetric = true)
dof_range = PRange(dof_partition)
elem_range = PRange(elem_partition)

IIs, JJs, VVs = tuple_of_arrays(map(meshes, partition(dof_range), ranks) do mesh, part, rank
    IIs, JJs, VVs = Int[], Int[], Float64[]
    dof_owners = local_to_owner(part)
    for conn in values(mesh.element_conns)
        global_conn = map(x -> part[x], conn)
        for e in axes(global_conn, 2)
            temp_global_conn = global_conn[:, e]
            for (local_row, global_row) in enumerate(temp_global_conn)
                if dof_owners[conn[local_row, e]] != rank
                    continue
                end

                for (local_col, global_col) in enumerate(temp_global_conn)
                    if dof_owners[conn[local_col, e]] != rank
                        continue
                    end
                    
                    push!(IIs, global_row)
                    push!(JJs, global_col)
                    push!(VVs, Ae[local_row, local_col])
                end
            end 
        end
    end
    return IIs, JJs, VVs
end)

map(meshes, partition(dof_range), ranks) do mesh, part, rank
    Is, Vs = Int[], Float64[]
    dof_owners = local_to_owner(part)
    ue = zeros(size(Ae,2))
    for conn in values(mesh.element_conns)
        global_conn = map(x -> part[x], conn)
        for e in axes(global_conn, 2)
            temp_global_conn = global_conn[:, e]
            fill!(ue, zero(eltype(ue)))
            for (local_node, global_node) in enumerate(temp_global_conn)

            end
        end
    end
end

# A = psparse(IIs, JJs, VVs, partition(dof_range), partition(dof_range))
# PartitionedArrays.psparse(identity, IIs, JJs, VVs, partition(dof_range), partition(dof_range))
