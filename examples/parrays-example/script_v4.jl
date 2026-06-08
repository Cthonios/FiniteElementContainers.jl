using Exodus
using FiniteElementContainers
using IterativeSolvers
using Krylov
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
dof     = DofManager(u, dbcs, mesh_file)
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

cg_workspace = CgWorkspace(K, -R)
Krylov.cg!(cg_workspace, K, -R)
x_new, stats = Krylov.results(cg_workspace)

U = create_field(dof)
# update_field_dirichlet_bcs!(U, dof)
update_field_unknowns!(U, dof, x_new)

pp = PostProcessor(output_file, mesh, true; extra_nodal_names = [names(u)...])
write_times(pp, 1, 0.0)
write_field(pp, 1, ["u"], U)
close(pp)
