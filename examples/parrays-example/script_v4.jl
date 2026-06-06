using Exodus
using FiniteElementContainers
using IterativeSolvers
using LinearAlgebra
using PartitionedArrays
using StaticArrays

include("../../test/poisson/TestPoissonCommon.jl")
f(X, _) = 2. * π^2 * sin(2π * X[1]) * sin(2π * X[2])
bc_func(_, _) = 0.
mesh_file = Base.source_dir() * "/square.g"
output_file = Base.source_dir() * "/output.e"

num_dofs = 1
num_ranks = 4
distribute = identity
# distribute = distribute_with_mpi
ranks = distribute(LinearIndices((num_ranks,)))

mesh    = UnstructuredMesh(mesh_file, num_ranks, ranks)
V       = FunctionSpace(mesh, H1Field, Lagrange)
physics = Poisson(f)
props   = create_properties(physics)
u       = ScalarFunction(V, "u")
dbcs    = DirichletBC[
    DirichletBC("u", bc_func; nodeset_name = "boundary")
]
dof     = DofManager(u, dbcs, num_ranks, ranks, mesh_file, mesh)
asm     = SparseMatrixAssembler(dof)
p       = create_parameters(mesh, asm, physics, props; dirichlet_bcs = dbcs)

# TODO
# first setup a solution vector by doing IterativeSolvers.cg(K, -0)
# so that we get a solution vector with the right sizes
# then we can copy that and cache them as a 
# current solution and increment

# clean up below
Uu = create_unknowns(asm)

U = create_field(asm)
update_field_unknowns!(U, dof, Uu)

assemble_stiffness!(asm, stiffness, U, p)
K = stiffness(asm)

assemble_vector!(asm, residual, U, p)
R = residual(asm)

x_new = IterativeSolvers.cg(K, -R, verbose = i_am_main(rank))

U = create_field(dof)
# update_field_dirichlet_bcs!(U, dof)
update_field_unknowns!(U, dof, x_new)

map(partition(U), dof.var.fspace, mesh, ranks) do U_local, fspace, mesh_local, rank
    u_temp = ScalarFunction(fspace, "u")
    output_file_temp = output_file * ".$(num_ranks).$(rank - 1)"
    pp = PostProcessor(mesh_local, output_file_temp, u_temp)
    write_times(pp, 1, 0.0)
    write_field(pp, 1, ["u"], U_local)
    close(pp)
end

map_main(ranks) do rank
    epu(output_file * ".$(num_ranks).$(rank - 1)")
end
