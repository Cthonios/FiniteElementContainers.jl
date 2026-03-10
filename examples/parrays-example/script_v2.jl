# using Exodus
using FiniteElementContainers
using IterativeSolvers
using LinearAlgebra
using PartitionedArrays

include("../../test/poisson/TestPoissonCommon.jl")
f(X, _) = 2. * π^2 * cos(π * X[1]) * cos(π * X[2])
u(x) = 0.5 * (x[1] + x[2])
h = 0.1
Ae = (h^2 / 6) * [
     4.0  -1.0  -1.0  -2.0
    -1.0   4.0  -2.0  -1.0
    -1.0  -2.0   4.0  -1.0
    -2.0  -1.0  -1.0   4.0
]
bc_func(_, _) = 0.
mesh_file = Base.source_dir() * "/square.g"
output_file = Base.source_dir() * "/output.e"
num_dofs = 1
num_ranks = 4
distribute = identity
# distribute = distribute_with_mpi

ranks = distribute(LinearIndices((num_ranks,)))

# serial problem setup (TODO remove this requirement and make it hidden to user)
smesh = UnstructuredMesh(mesh_file)
V = FunctionSpace(smesh, H1Field, Lagrange)
uvar = ScalarFunction(V, :u)
sdof = DofManager(uvar)
dbcs = DirichletBC[
    DirichletBC(:u, bc_func; nodeset_name = :boundary),
]
dbcs = DirichletBCs(smesh, sdof, dbcs)
boundary_dofs = dirichlet_dofs(dbcs)

# start parallel stuff
meshes = UnstructuredMesh(mesh_file, num_ranks, ranks)

# create partitions
parts = FiniteElementContainers.create_partition(mesh_file, num_dofs, num_ranks, ranks, boundary_dofs)
(; elem_parts, field_parts, unknown_parts, field_to_unknown, field_colors, unknown_to_field) = parts

mat_pattern = FiniteElementContainers.create_matrix_sparsity_pattern(meshes, parts)

VVs = map(meshes, parts.elem_parts, ranks) do mesh, part, rank
    VVs = Float64[]
    for (key, conn) in mesh.element_conns
        for e in axes(conn, 2)
            for i in axes(conn, 1)
                for j in axes(conn, 1)
                    push!(VVs, Ae[i, j])
                end
            end
        end
    end
    VVs
end

# display(VVs)

# original way for matrices
# IIs, JJs, VVs = tuple_of_arrays(map(meshes, elem_parts, ranks) do mesh, part, rank
#     IIs, JJs, VVs = Int[], Int[], Float64[]

#     for (key, conn) in mesh.element_conns
#         for e in axes(conn, 2)
#             glob_conn = @views mesh.node_id_map[conn[:, e]]
#             for i in axes(glob_conn, 1)
#                 if insorted(glob_conn[i], boundary_dofs)
#                     continue
#                 end

#                 for j in axes(glob_conn, 1)
#                     if insorted(glob_conn[j], boundary_dofs)
#                         continue
#                     end

#                     push!(IIs, field_to_unknown[glob_conn[i]])
#                     push!(JJs, field_to_unknown[glob_conn[j]])
#                     push!(VVs, Ae[i, j])
#                 end
#             end
#         end
#     end

#     IIs, JJs, VVs
# end)


# temp_vals = map(mat_pattern.unknown_dofs, VVs) do dofs, val
#     val[dofs]
# end
# display(temp_vals)

Is, Vs = tuple_of_arrays(map(meshes, elem_parts, ranks) do mesh, part, rank
    Is, Vs = Int[], Float64[]
    ue = zeros(size(Ae, 2))
    ge = similar(ue)
    X = mesh.nodal_coords
    for (key, conn) in mesh.element_conns
        for e in axes(conn, 2)
            glob_conn = @views mesh.node_id_map[conn[:, e]]
            fill!(ue, 0.0)

            for i in axes(glob_conn, 1)
                if insorted(glob_conn[i], boundary_dofs)
                    ue[i] = u(X[:, conn[i, e]])
                end
            end

            mul!(ge, Ae, ue)
            for i in axes(glob_conn, 1)
                if insorted(glob_conn[i], boundary_dofs)
                    continue
                end

                push!(Is, field_to_unknown[glob_conn[i]])
                push!(Vs, -ge[i])
            end
        end
    end 
    return Is, Vs
end)

# A = psparse(IIs, JJs, VVs, unknown_parts, unknown_parts) |> fetch
A = psparse(mat_pattern, VVs) |> fetch
b = pvector(Is, Vs, unknown_parts) |> fetch
x = IterativeSolvers.cg(A, b, verbose = i_am_main(rank))

u_an_unknown = map(meshes, unknown_parts, ranks) do mesh, part, rank
    X = mesh.nodal_coords
    us = Float64[]
    for n in part
        nfield = unknown_to_field[n]
        push!(us, u(smesh.nodal_coords[:, nfield]))
    end
    us
end

u_field = map(meshes, unknown_parts, ranks, x.index_partition, x.vector_partition) do mesh, part, rank, xpart, xvec
    V = FunctionSpace(mesh, H1Field, Lagrange)
    uvar = ScalarFunction(V, :u)
    asm = SparseMatrixAssembler(uvar)
    physics = Poisson(f)
    props = create_properties(physics)
    dbcs = DirichletBC[
        DirichletBC(:u, bc_func; nodeset_name = :boundary),
    ]
    p = create_parameters(mesh, asm, physics, props; dirichlet_bcs=dbcs)
    U = create_field(asm)

    for (n, val) in zip(xpart, xvec)
        nfield = unknown_to_field[n]
        index = indexin(nfield, mesh.node_id_map)[1]
        if index !== nothing
            U[index] = val
        end
    end
    U
end
u_field = PVector(u_field, field_parts)
display(partition(u_field))
display(own_values(x) .≈ u_an_unknown)
# @assert all(own_values(x) .≈ u_an_unknown)