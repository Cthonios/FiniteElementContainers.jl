using Exodus
using FiniteElementContainers
using PartitionedArrays
using Test

function test_decomp_to_epu()
    mesh_file = dirname(Base.source_dir()) * "/poisson/poisson.g"
    num_ranks = 4
    FiniteElementContainers.decompose_mesh(mesh_file, num_ranks, 1)

    new_mesh_file = dirname(Base.source_dir()) * "/poisson/temp_mesh.g"

    for n in 1:num_ranks
        @test isfile(mesh_file * ".$num_ranks.$(n - 1)")
    end
    @test isfile(mesh_file * ".nem")
    @test isfile(mesh_file * ".pex")

    for n in 1:num_ranks
        cp(
            mesh_file * ".$num_ranks.$(n - 1)",
            new_mesh_file * ".$num_ranks.$(n - 1)";
            force = true
        )
    end

    Exodus.epu(new_mesh_file)
    temp = split(new_mesh_file, "/")[end]
    @test Exodus.exodiff(mesh_file, String(temp))

    # cleanup
    for n in 1:num_ranks
        rm(mesh_file * ".$num_ranks.$(n - 1)"; force = true)
        rm(new_mesh_file * ".$num_ranks.$(n - 1)"; force = true)
    end
    rm(mesh_file * ".nem"; force = true)
    rm(mesh_file * ".pex"; force = true)
    rm(temp; force = true)
    rm(dirname(mesh_file) * "/decomp.log"; force = true)
    rm(dirname(temp) * "/epu.log"; force = true)
    rm(dirname(dirname(mesh_file)) * "/epu_err.log"; force = true)
    rm(dirname(dirname(mesh_file)) * "/exodiff.log"; force = true)
end

# function test_global_colors(num_dofs, num_ranks)
#     mesh_file = dirname(Base.source_dir()) * "/poisson/poisson.g"
#     FiniteElementContainers.decompose_mesh(mesh_file, num_ranks, 1)
#     # global_colorings = FiniteElementContainers.global_colorings(mesh_file, num_dofs, num_ranks)

    

#     exo = ExodusDatabase(mesh_file, "r")
#     n_nodes = Exodus.initialization(exo).num_nodes
#     # make sure it's the right length
#     @test length(global_colorings) == n_nodes * num_dofs
#     # ensure each proc has a sensible number
#     for n in axes(global_colorings, 1)
#         @test global_colorings[n] >= 1
#         @test global_colorings[n] <= num_ranks
#     end

#     # cleanup
#     for n in 1:num_ranks
#         rm(mesh_file * ".$num_ranks.$(n - 1)"; force = true)
#     end
#     rm(mesh_file * ".nem"; force = true)
#     rm(mesh_file * ".pex"; force = true)
#     rm(dirname(mesh_file) * "/decomp.log"; force = true)
# end

# function test_variable_partition(distribute, num_dofs, num_ranks)
#     # mesh_file = dirname(Base.source_dir()) * "/poisson/poisson.g"
#     mesh_file = Base.source_dir() * "/square.g"
#     FiniteElementContainers.decompose_mesh(mesh_file, num_ranks, num_dofs)
#     global_colorings = FiniteElementContainers.global_colorings(mesh_file, num_dofs, num_ranks)

#     ranks = distribute(LinearIndices((num_ranks,)))
#     meshes = map(ranks) do rank
#         UnstructuredMesh(mesh_file, num_ranks, rank)
#     end
#     parts = FiniteElementContainers.create_partition(mesh_file, num_dofs, num_ranks, ranks)

#     # test that the partitions owned and ghost members are the right size
#     map(meshes, parts.parts, ranks) do mesh, part, rank
#         node_map = mesh.node_id_map
#         num_dofs_in_rank = count(x -> x == rank, global_colorings)

#         n_local = length(part.ranges[1])
#         n_ghost = length(part.ghost.ghost_to_owner)
#         @test n_local == num_dofs_in_rank
#         @test n_local + n_ghost == length(node_map)
#     end

#     # now test that the exo_to_par and par_to_exo maps are consistent
#     exo_to_par, par_to_exo = parts.exo_to_par, parts.par_to_exo

#     for n in 1:length(global_colorings)
#         @test n in values(exo_to_par)
#         @test n in values(par_to_exo)
#     end

#     par_node_maps = map(meshes, parts.parts) do mesh, part
#         node_map = mesh.node_id_map
#         # n_ghost = length(part.ghost.ghost_to_owner)
#         # n_local = length(part.ranges[1])
#         # ranges = part.ranges[1]

#         par_node_map = map(x -> exo_to_par[x], node_map)
#         exo_node_map = map(x -> par_to_exo[x], par_node_map)

#         # checking they're invertible maps
#         for n in axes(node_map, 1)
#             @test node_map[n] == exo_node_map[n]
#             @test node_map[n] == node_map[par_to_exo[exo_to_par[n]]]
#         end

#         # NOTE this below is dumb.
#         # we shouldn't expect node id maps to be ordered in terms
#         # of local/ghost since then exodus would have an implicit local/global
#         # which it does not.
#         # instead we should add tests that test things based on load balance info
#         # @test all(part.ranges[1] |> collect .== part[1:n_local])
#         # @show "hur"
#         # # @test all(part.ranges[1] |> collect .== par_node_map[1:n_local])
#         # # par_node_map
#         # parts

#         # test connectivities are invertible
#         exo_conns = mesh.element_conns
#         par_conns = deepcopy(exo_conns)
#         for (block, conn) in enumerate(values(par_conns))
#             for e in axes(conn, 2)
#                 for n in axes(conn, 1)
#                     conn[n, e] = exo_to_par[node_map[conn[n, e]]]
#                 end
#             end

#             # TODO this isn't quite right yet
#             # # now invert real quick
#             # temp_conns = deepcopy(conn)
#             # for e in axes(temp_conns, 2)
#             #     for n in axes(temp_conns, 1)
#             #         temp_conns[n, e] = par_to_exo[conn[n, e]]
#             #     end
#             # end

#             # @test all(values(exo_conns)[block] .== values(temp_conns))
#         end
#     end

#     # cleanup
#     for n in 1:num_ranks
#         rm(mesh_file * ".$num_ranks.$(n - 1)"; force = true)
#     end
#     rm(mesh_file * ".nem"; force = true)
#     rm(mesh_file * ".pex"; force = true)
#     rm(dirname(mesh_file) * "/decomp.log"; force = true)
#     # parts
#     par_node_maps
# end

if !Sys.iswindows()
    test_decomp_to_epu()
end
# test_global_colors(1, 4)
# test_global_colors(2, 4)
# test_global_colors(3, 4)
# test_variable_partition(identity, 1, 4)
