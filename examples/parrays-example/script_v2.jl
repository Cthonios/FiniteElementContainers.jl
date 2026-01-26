using Exodus
using FiniteElementContainers
using IterativeSolvers
using LinearAlgebra
using PartitionedArrays

include("../../test/poisson/TestPoissonCommon.jl")
f(X, _) = 2. * π^2 * cos(π * X[1]) * cos(π * X[2])

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

exo_parts = FiniteElementContainers.create_partition(
    mesh_file, num_dofs, num_ranks, ranks;
    add_element_borders = false,
    add_ghost_nodes = true
)

sol_parts = variable_partition(exo_parts.ndpp, sum(exo_parts.ndpp))
X = pzeros(sol_parts)

asms, U_data, ps = tuple_of_arrays(map(exo_parts.dof_parts, ranks) do part, rank
    mesh = UnstructuredMesh(mesh_file, num_ranks, rank)
    V = FunctionSpace(mesh, H1Field, Lagrange)
    physics = Poisson(f)
    props = create_properties(physics)
    u = ScalarFunction(V, :u)
    asm = SparseMatrixAssembler(u; use_condensed = true)
    U = create_field(asm).data
    p = create_parameters(mesh, asm, physics, props)
    asm, U, p
end)

# this is a "field" in FiniteElementContainers parlance
U = PVector(U_data, exo_parts.dof_parts)

# for n in 1:5
    # how to update U given X
    map(partition(U), partition(X)) do u, x
        for n in axes(x, 1)
            u[n] = x[n]
        end
        u
    end

    Rs, Ks = tuple_of_arrays(map(asms, partition(U), ps) do asm, u, p
        assemble_vector!(asm.residual_storage, asm.dof, residual, u, p)
        assemble_matrix!(asm.stiffness_storage, asm.matrix_pattern, asm.dof, stiffness, u, p)
        residual(asm), stiffness(asm)
    end)

    R = PVector(Rs, exo_parts.dof_parts)
    assemble!(R) |> wait
    R = PVector(own_values(R), sol_parts) |> fetch


    Is, Js, Vs = tuple_of_arrays(map(exo_parts.dof_parts, asms, ranks) do part, asm, rank
        # need to sort throught he assembled values and indices
        # adn push to new arrays to make sure they're owned
        # by this rank
        Is, Js, Vs = Int[], Int[], Float64[]
        par_to_exo = exo_parts.dof_par_to_exo

        exo_dof_to_own = exo_parts.exo_dof_to_own
        mat_pattern = asm.matrix_pattern
        loc_to_glob = local_to_global(part)
        for (i, j, v) in zip(mat_pattern.Is, mat_pattern.Js, asm.stiffness_storage)
            # @show local_to_global(part)
            i_glob = loc_to_glob[i]
            j_glob = loc_to_glob[j]
            i_own = exo_dof_to_own[par_to_exo[i_glob]]
            j_own = exo_dof_to_own[par_to_exo[j_glob]]

            # if i_own == rank && j_own == rank
            #     # continue
            # # else
            #     push!(Is, i_glob)
            #     push!(Js, j_glob)
            #     push!(Vs, v)
            # end
            if (i in own_to_local(part)) && (j in own_to_local(part))
                push!(Is, i_glob)
                push!(Js, j_glob)
                push!(Vs, v)
            end
        end
        return Is, Js, Vs
    end)

    K = psparse(Is, Js, Vs, exo_parts.dof_parts, exo_parts.dof_parts) |> fetch
    # K = psparse(Is, Js, Vs, sol_parts, sol_parts) |> fetch

    # K = psparse(laplacian_fem((121,), (4,), ranks)...; assembled = true) |> fetch
    # b = K * X
    # x = IterativeSolvers.cg(K, -R; verbose = true)

    # K_data = map(exo_parts.dof_parts, Ks) do part, K
    #     # part
    #     # own_ids = 
    #     own_ids = own_to_local(part)
    #     ghost_ids = ghost_to_local(part)
    #     own_own = K[own_ids, own_ids]
    #     own_ghost = K[own_ids, ghost_ids]
    #     ghost_own = K[ghost_ids, own_ids]
    #     ghost_ghost = K[ghost_ids, ghost_ids]
    #     mat_part = split_matrix_blocks(own_own, own_ghost, ghost_own, ghost_own)
    #     # row_perm = 1:(length(own_ids))
    #     SplitMatrix(mat_part, own_ids, own_ids)
    # end

    # K = PSparseMatrix(K_data, sol_parts, sol_parts, true)
    # assemble(K)
    # x = IterativeSolvers.cg(K, R; verbose = true)
# end

# Is, Vs = tuple_of_arrays(map(exo_parts.elem_parts, exo_parts.par_conns, ranks) do part, conns, rank
#     Is = Int[]
#     for elem in part
#         # if elem_to_own[elem_par_to_exo[elem]] != rank
#         #     continue
#         # end

#         for node in exo_parts.exo_elem_to_exo_node[exo_parts.elem_par_to_exo[elem]]
#             par_node = exo_parts.dof_exo_to_par[node]
#             if exo_parts.exo_dof_to_own[node] == rank
#                 push!(Is, par_node)
#             end
#         end
#     end
#     Is, zeros(length(Is))
# end)

# IIs, JJs, VVs = tuple_of_arrays(map(exo_parts.elem_parts, exo_parts.par_conns, ranks) do part, conns, rank
#     IIs, JJs = Int[], Int[]
#     h = 0.1
#     Ae = (h^2/6) * [
#         4.0  -1.0  -1.0  -2.0
#         -1.0   4.0  -2.0  -1.0
#         -1.0  -2.0   4.0  -1.0
#         -2.0  -1.0  -1.0   4.0
#     ]

#     for elem in part
#         if exo_parts.exo_elem_to_own[exo_parts.elem_par_to_exo[elem]] != rank
#             continue
#         end
        
#         nodes = exo_parts.exo_elem_to_exo_node[exo_parts.elem_par_to_exo[elem]]
#         for i in nodes
#             for j in nodes
#                 par_i = exo_parts.dof_exo_to_par[i]
#                 par_j = exo_parts.dof_exo_to_par[j]
#                 push!(IIs, par_i)
#                 push!(JJs, par_j)
#             end
#         end
#     end
#     return IIs, JJs, zeros(length(IIs))
# end)

# function solve(asms, Us, ps)
#     x = pones(exo_parts.dof_parts) |> fetch
#     for n in 1:10
#         Rvs, Kvs = tuple_of_arrays(map(asms, Us, ps) do asm, U, p
#             assemble_vector!(asm.residual_storage, asm.dof, residual, U, p)
#             assemble_matrix!(asm.stiffness_storage, asm.matrix_pattern, asm.dof, stiffness, U, p)
#             asm.residual_storage, asm.stiffness_storage
#         end)
#         b = pvector(Is, Rvs, exo_parts.dof_parts) |> fetch
#         A = psparse(IIs, JJs, Kvs, exo_parts.dof_parts, exo_parts.dof_parts) |> fetch
#         # x = pzeros(dof_parts) |> fetch
#         dx = IterativeSolvers.cg(A, b, verbose=false)
#         x = x + dx

#         Us = map(partition(x), Us) do part, U
#             # @show size(U)
#             # @show size(part)
#             # U[1, :] .= part[1:36]
#             U.data[1:length(part)] .= part
#             U
#         end

#         @show norm(b)
#         @show norm(x)
#     end
# end

# solve(asms, Us, ps)
# map(meshes, us, Us, ranks) do mesh, u, U, rank
#     og_mesh_file = mesh_file * ".$(num_ranks).$(rank - 1)"
#     output_file = "output-file.e.$(num_ranks).$(rank - 1)"
#     pp = PostProcessor(mesh, output_file, u)
#     write_times(pp, 1, 0.0)
#     write_field(pp, 1, ("u",), H1Field(reshape(U[1:36], 1, length(U[1:36]))))
#     close(pp)
# end
