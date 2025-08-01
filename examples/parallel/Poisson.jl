import FiniteElementContainers: ExodusMesh
using Exodus
using FiniteElementContainers
using IterativeSolvers
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

local_idss = map(node_id_maps, parts) do node_id_map, part
    ids = local_to_global(part)
    ids = indexin(node_id_map, part) |> sort
    ids
end

asms = map(parts, ranks) do part, rank
    mesh_file_rank = "$mesh_file.$np.$(rank - 1)"
    mesh = UnstructuredMesh(ExodusMesh, mesh_file_rank, false, false)
    V = FunctionSpace(mesh, H1Field, Lagrange)
    u = ScalarFunction(V, :u)
    dof = DofManager(u)
    asm = SparseMatrixAssembler(dof, H1Field)
end

ps = map(asms) do asm 
    # dbcs = DirichletBC[
    #     DirichletBC(:u, :sset_1, bc_func),
    #     DirichletBC(:u, :sset_2, bc_func),
    #     DirichletBC(:u, :sset_3, bc_func),
    #     DirichletBC(:u, :sset_4, bc_func),
    # ]
    # dbcs = DirichletBC[]
    physics = Poisson()
    props = create_properties(physics)
    p = create_parameters(asm, physics, props)
end

Uus = pzeros(parts)
Rs = pzeros(parts)

# try some assembly
Rs_new = map(partition(Rs), asms, local_idss, partition(Uus), ps) do R, asm, local_ids, Uu, p
    Uu_temp = Uu[local_ids]
    assemble_vector!(asm, Uu_temp, p, H1Field, residual)
    R[local_ids] .= residual(asm)
    R
end

IJV = map(asms, local_idss, partition(Uus), ps) do asm, local_ids, Uu, p
    Uu_temp = Uu[local_ids]
    assemble_matrix!(asm, Uu_temp, p, H1Field, stiffness)
    asm.pattern.Is, asm.pattern.Js, asm.stiffness_storage 
end
I, J, V = tuple_of_arrays(IJV)

b = PVector(Rs_new, parts) |> fetch
A = psparse(I, J, V, parts, parts) |> fetch
x = similar(b, axes(A, 2))
x .= b
IterativeSolvers.cg!(x, A, -b, verbose=true)
x