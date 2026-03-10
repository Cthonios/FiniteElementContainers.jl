using Exodus
using FiniteElementContainers
using Gmsh

p_order = 2
# mesh_file = dirname(dirname(Base.source_dir())) * "/test/poisson/poisson.g"
# mesh_file = dirname(dirname(Base.source_dir())) * "/test/poisson/multi_block_mesh_quad4_tri3.g"
mesh_file = dirname(dirname(Base.source_dir())) * "/test/gmsh/square_meshed_with_tris.geo"
# output_file = "output-poisson-neumann-$(p_order).e"
output_file = "output-gmsh-test.e"
# f(_, _) = 1."
# f(X, _) = 2. * π^2 * cos(2π * X[1]) * cos(2π * X[2])
bc_func(_, _) = 0.
f(X, _) = 2. * π^2 * sin(2π * X[1]) * sin(2π * X[2])


include("../../test/poisson/TestPoissonCommon.jl")

# mesh = UnstructuredMesh(mesh_file; p_order = p_order)
mesh = UnstructuredMesh(mesh_file)
V = FunctionSpace(mesh, H1Field, Lagrange) 
physics = Poisson(f)
props = create_properties(physics)
u = ScalarFunction(V, :u)
asm = SparseMatrixAssembler(u; use_condensed = true)

dbcs = DirichletBC[
    DirichletBC(:u, bc_func; nodeset_name = :boundary)
]
# dbcs = nothing

p = create_parameters(mesh, asm, physics, props; dirichlet_bcs = dbcs)
solver = NewtonSolver(DirectLinearSolver(asm))
integrator = QuasiStaticIntegrator(solver)
evolve!(integrator, p)

U = p.h1_field

# pp = PostProcessor(mesh, output_file, u; copy_mesh_file = false)
pp = PostProcessor(mesh, output_file, u)
write_times(pp, 1, 0.0)
write_field(pp, 1, ("u",), U)
close(pp)
