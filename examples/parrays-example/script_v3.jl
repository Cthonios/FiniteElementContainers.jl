using FiniteElementContainers
using IterativeSolvers
using LinearAlgebra
using PartitionedArrays

include("../../test/poisson/TestPoissonCommon.jl")
f(X, _) = 2. * π^2 * cos(π * X[1]) * cos(π * X[2])
u(x) = 0.5 * (x[1] + x[2])
h = 0.1
# Ae = (h^2 / 6) * [
#      4.0  -1.0  -1.0  -2.0
#     -1.0   4.0  -2.0  -1.0
#     -1.0  -2.0   4.0  -1.0
#     -2.0  -1.0  -1.0   4.0
# ]
Ae = (1 / 6) * [
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
vec_pattern = FiniteElementContainers.create_vector_sparsity_pattern(meshes, parts)

Xs = nodal_coordinates(meshes, parts)

VVs = map(meshes, parts.elem_parts) do mesh, part
    VVs = Float64[]
    Ae_vec = vec(Ae)
    for (key, conn) in mesh.element_conns
        for e in axes(conn, 2)
            for j in axes(conn, 1)
                for i in axes(conn, 1)
                    push!(VVs, Ae[i, j])
                end
            end
        end
    end
    VVs
end

Vs = map(meshes, partition(Xs)) do mesh, X
    Vs = Float64[]
    ue = zeros(size(Ae, 2))
    ge = similar(ue)
    # X = mesh.nodal_coords

    for (key, conn) in mesh.element_conns
        for e in axes(conn, 2)
            glob_conn = @views mesh.node_id_map[conn[:, e]]
            fill!(ue, 0.0)

            for i in axes(glob_conn, 1)
                # this is for bcs here
                if insorted(glob_conn[i], boundary_dofs)
                    ue[i] = u(X[:, conn[i, e]])
                end
            end

            mul!(ge, Ae, ue)

            for i in axes(glob_conn, 1)
                push!(Vs, -ge[i])
            end
        end
    end
    Vs
end

# A = psparse(mat_pattern, VVs) |> fetch
# b = pvector(vec_pattern, Vs) |> fetch
# x = IterativeSolvers.cg(A, b, verbose = i_am_main(rank))

# Uu = pzeros(vec_pattern)
Uus = create_unknowns(parts)
# U = create_field(parts)

# X_vals = map(meshes) do mesh
#     mesh.nodal_coords
# end
# Xs = PVector(X_vals, field_parts)

# let's just try setting up stiffness first...
V2s, VV2s = tuple_of_arrays(map(meshes, partition(Uus), mat_pattern.unknown_dofs) do mesh, Uu, indices
    V = FunctionSpace(mesh, H1Field, Lagrange)
    uvar = ScalarFunction(V, :u)
    asm = SparseMatrixAssembler(uvar)
    physics = Poisson(f)
    props = create_properties(physics)
    dbcs = DirichletBC[
        DirichletBC(:u, bc_func; nodeset_name = :boundary)
    ]
    p = create_parameters(mesh, asm, physics, props; dirichlet_bcs = dbcs)
    U = create_field(asm)
    Uu = create_unknowns(asm)
    assemble_vector!(asm, residual, Uu, p)
    assemble_stiffness!(asm, stiffness, Uu, p)
    asm.residual_storage.data, asm.stiffness_storage
end)

# A = psparse(mat_pattern, VVs) |> fetch
# # b = pzeros(vec_pattern) |> fetch
# b = pvector(vec_pattern, Vs) |> fetch
A = psparse(mat_pattern, VV2s) |> fetch
b = pvector(vec_pattern, V2s) |> fetch
x = IterativeSolvers.cg(A, b, verbose = i_am_main(rank))
