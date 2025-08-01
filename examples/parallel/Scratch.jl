import FiniteElementContainers: ExodusMesh
using Exodus
using FiniteElementContainers
using PartitionedArrays

# eventually command line options
np = 4
n_dofs = 1
ranks = LinearIndices((np,))

mesh_file = "test/poisson/poisson.g"
output_file = "output.e"

# global funcs
# f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
f(X, _) = 1.
bc_func(_, _) = 0.
include("../../test/poisson/TestPoissonCommon.jl")

# setup partitions, not exactly meshes
# using some name piracy here
node_id_maps, parts = UnstructuredMesh(mesh_file, ranks, np, n_dofs)


# asms = map(parts, ranks) do part, rank
#     mesh_file_rank = "$mesh_file.$np.$(rank - 1)"
#     mesh = UnstructuredMesh(ExodusMesh, mesh_file_rank, false, false)
#     V = FunctionSpace(mesh, H1Field, Lagrange)
#     @show coords = mesh.nodal_coords |> size
#     u = ScalarFunction(V, :u)
#     dof = DofManager(u)
#     asm = SparseMatrixAssembler(dof, H1Field)
# end

# ps = map(asms) do asm 
#     # dbcs = DirichletBC[
#     #     DirichletBC(:u, :sset_1, bc_func),
#     #     DirichletBC(:u, :sset_2, bc_func),
#     #     DirichletBC(:u, :sset_3, bc_func),
#     #     DirichletBC(:u, :sset_4, bc_func),
#     # ]
#     # dbcs = DirichletBC[]
#     physics = Poisson()
#     props = create_properties(physics)
#     p = create_parameters(asm, physics, props)
# end

# setup passembler
# Uus = pzeros(parts)

# IV = map(asms, parts, own_values(Uus), ps) do asm, part, Uu, p
#     # Is = local_to_global(part)
#     # Vs = zeros(length(Is))
#     # Is, Vs
#     # @show size(Uu)
#     # @show asm
#     # FiniteElementContainers.assemble!(asm, Uu, p, Val{:residual}(), H1Field)

#     part.own.own_to_global |> length
# end

# map(asms) do asm
#     fspace = FiniteElementContainers.function_space(asm, H1Field)
#     fspace.elem_conns.block_1 |> maximum
# end

# trying something simple
x = pzeros(parts)
y = pzeros(parts)
# owns = own_to_local.(parts)
# ghosts = ghost_to_local.(parts)
# idss = vcat.(owns, ghosts)

# map(node_id_maps, parts, ranks) do node_id_map, part, rank
#     mesh_file_rank = "$mesh_file.$np.$(rank - 1)"
#     exo = ExodusDatabase(mesh_file_rank, "r")
#     ids = local_to_global(part)
#     ids = indexin(node_id_map, part)
#     display(ids)
# end

x_parts = map(node_id_maps, parts, ranks, partition(x)) do node_id_map, part, rank, v_part
    mesh_file_rank = "$mesh_file.$np.$(rank - 1)"
    exo = ExodusDatabase(mesh_file_rank, "r")
    coords = read_coordinates(exo)
    n_nodes = exo.init.num_nodes
    close(exo)

    ids = local_to_global(part)
    ids = indexin(node_id_map, part) |> sort
    v_part[ids] = coords[1, :]
    v_part
end

x = PVector(x_parts, parts)
# consistent!(x) |> wait

map(node_id_maps, parts, ranks, partition(x), partition(y)) do node_id_map, part, rank, x_part, y_part
    mesh_file_rank = "$mesh_file.$np.$(rank - 1)"
    output_file_rank = "output.e.$np.$(rank - 1)"
    copy_mesh(mesh_file_rank, output_file_rank)
    exo = ExodusDatabase(output_file_rank, "rw")
    n_nodes = exo.init.num_nodes
    write_names(exo, NodalVariable, ["x"])
    write_time(exo, 1, 0.0)

    ids = local_to_global(part)
    ids = indexin(node_id_map, part) |> sort
    write_values(exo, NodalVariable, 1, "x", x_part[ids])
    close(exo)
end

epu("output.e")

exo_1 = ExodusDatabase(mesh_file, "r")
exo_2 = ExodusDatabase("output.e", "r")
nmap_1 = read_id_map(exo_1, NodeMap)
nmap_2 = read_id_map(exo_2, NodeMap)
coords = read_coordinates(exo_1)
x = read_values(exo_2, NodalVariable, 1, "x")

@show coords[1, :] ≈ x[nmap_1]
coords[1, :] - x[nmap_1] |> sum