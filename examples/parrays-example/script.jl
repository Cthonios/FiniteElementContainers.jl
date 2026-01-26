using Exodus
using FiniteElementContainers
using IterativeSolvers
using PartitionedArrays

include("../../test/poisson/TestPoissonCommon.jl")
# f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
f(X, _) = 2. * π^2 * cos(2π * X[1]) * cos(2π * X[2])

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

parts = FiniteElementContainers.create_partition(mesh_file, num_dofs, num_ranks, ranks)

# need to grab a different connectivity numbering
# for setting up the global (e.g. has all ranks)
# assmbly stuff
par_conns = map(parts.dof_parts, ranks) do part, rank
    mesh = UnstructuredMesh(mesh_file, num_ranks, rank)
    conns = mesh.element_conns
    node_map = mesh.node_id_map
    # update in place since we'll make new mesh objects later
    for conn in values(conns)
        for e in axes(conn, 2)
            for n in axes(conn, 1)
                conn[n, e] = parts.exo_to_par[node_map[conn[n, e]]]
            end
        end
    end
    conns
end

nothing

# # now set up sparse partition stuff
# # TODO below will only work for one dof,
# # just to make things easy to start
# Is, Vs = tuple_of_arrays(map(par_conns, parts.dof_parts, ranks) do conns, part, rank
#     Is = Int[]
    
#     for block_conn in values(conns)
#         for e in axes(block_conn, 2)
#             conn = @views block_conn[:, e]
#             for node in conn
#                 # need to check for global coloring here
#                 # if parts.global_dof_colors[parts.par_to_exo[node]] == rank
#                     push!(Is, node)
#                 # end
#             end
#         end
#     end
#     Vs = zeros(length(Is))
#     return Is, Vs
# end)


# IIs, JJs, VVs = tuple_of_arrays(map(par_conns, parts.dof_parts, ranks) do conns, part, rank
#     IIs, JJs = Int[], Int[]

#     for block_conn in values(conns)
#         for e in axes(block_conn, 2)
#             conn = @views block_conn[:, e]
#             # for nodes in Iterators.product(conn, conn)
#             #     if parts.global_dof_colors[parts.par_to_exo[nodes[1]]] == rank
#             #         push!(IIs, nodes[1])
#             #     end
#             #     if parts.global_dof_colors[parts.par_to_exo[nodes[2]]] == rank
#             #         push!(JJs, nodes[2])
#             #     end
#             # end
#             # for node1 in conn
#             #     if parts.global_dof_colors[parts.par_to_exo[node1]] != rank
#             #         continue
#             #     end
#             #     for node2 in conn
#             #         if parts.global_dof_colors[parts.par_to_exo[node2]] != rank
#             #             continue
#             #         end
#             #         push!(IIs, node1)
#             #         push!(JJs, node2)
#             #     end
#             # end
#             for node1 in conn
#                 if parts.global_dof_colors[parts.par_to_exo[node1]] != rank
#                     continue
#                     # push!(IIs, nodes[1])
#                 end
#                 for node2 in conn
#                     # if parts.global_dof_colors[parts.par_to_exo[node2]] != rank
#                     #     continue
#                     #     # push!(IIs, nodes[1])
#                     # end
#                     push!(IIs, node1)
#                     push!(JJs, node2)
#                 end
#             end
#         end
#     end
#     VVs = zeros(length(IIs))
#     return IIs, JJs, VVs
# end)

# meshes, us, Rvs, Kvs = tuple_of_arrays(map(parts.dof_parts, ranks) do part, rank
#     mesh = UnstructuredMesh(mesh_file, num_ranks, rank)
#     V = FunctionSpace(mesh, H1Field, Lagrange)
#     physics = Poisson(f)
#     props = create_properties(physics)
#     u = ScalarFunction(V, :u)
#     asm = SparseMatrixAssembler(u; use_condensed=true)
#     Rv = asm.residual_storage
#     Kv = asm.stiffness_storage
#     U = create_field(asm)
#     p = create_parameters(mesh, asm, physics, props)
#     assemble_vector!(Rv, asm.dof, residual, U, p)
#     assemble_matrix!(Kv, asm.matrix_pattern, asm.dof, stiffness, U, p)
#     mesh, u, Rv, Kv
# end)

# A = psparse(IIs, JJs, Kvs, parts.dof_parts, parts.dof_parts) |> fetch
# b = pvector(Is, Rvs, parts.dof_parts) |> fetch

# x = similar(b, axes(A, 2))
# x .= b
# IterativeSolvers.cg!(x, A, b; verbose=true)
# x

# map(meshes, us, local_values(x), ranks) do mesh, u, U, rank
#     og_mesh_file = mesh_file * ".$(num_ranks).$(rank - 1)"
#     output_file = "output-file.e.$(num_ranks).$(rank - 1)"
#     pp = PostProcessor(mesh, output_file, u)
#     write_times(pp, 1, 0.0)
#     write_field(pp, 1, ("u",), H1Field(reshape(U, 1, length(U))))
#     close(pp)
# end
