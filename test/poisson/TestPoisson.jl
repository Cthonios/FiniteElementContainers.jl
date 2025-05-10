using Exodus
using FiniteElementContainers

# mesh file
gold_file = "./poisson/poisson.gold"
mesh_file = "./poisson/poisson.g"
output_file = "./poisson/poisson.e"

# methods for a simple Poisson problem
f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
bc_func(_, _) = 0.

include("TestPoissonCommon.jl")

# read mesh and relevant quantities

function poisson()
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Poisson()
  u = ScalarFunction(V, :u)
  asm = SparseMatrixAssembler(H1Field, u)
  pp = PostProcessor(mesh, output_file, u)

  # setup and update bcs
  dbcs = DirichletBC[
    DirichletBC(:u, :sset_1, bc_func),
    DirichletBC(:u, :sset_2, bc_func),
    DirichletBC(:u, :sset_3, bc_func),
    DirichletBC(:u, :sset_4, bc_func),
  ]

  # setup the parameters
  p = create_parameters(asm, physics; dirichlet_bcs=dbcs)

  # setup solver and integrator
  solver = NewtonSolver(IterativeLinearSolver(asm, :CgSolver))
  integrator = QuasiStaticIntegrator(solver)
  evolve!(integrator, p)

  write_times(pp, 1, 0.0)
  write_field(pp, 1, p.h1_field)
  close(pp)

  if !Sys.iswindows()
    @test exodiff(output_file, gold_file)
  end
  rm(output_file; force=true)
end

@time poisson()
# @time poisson()

# # condensed test
# mesh = UnstructuredMesh(mesh_file)
# V = FunctionSpace(mesh, H1Field, Lagrange) 
# physics = Poisson()
# u = ScalarFunction(V, :u)
# asm = SparseMatrixAssembler(H1Field, u)
# # pp = PostProcessor(mesh, output_file, u)

# # setup and update bcs
# dbcs = DirichletBC[
#   DirichletBC(asm.dof, :u, :sset_1, bc_func),
#   DirichletBC(asm.dof, :u, :sset_2, bc_func),
#   DirichletBC(asm.dof, :u, :sset_3, bc_func),
#   DirichletBC(asm.dof, :u, :sset_4, bc_func),
# ]
# update_dofs!(asm, dbcs; use_condensed=true)
# Uu = create_unknowns(asm)
# Ubc = create_bcs(asm, H1Field)
# U = create_field(asm, H1Field)
# update_field!(U, asm, Uu, Ubc)
# update_field_bcs!(U, asm.dof, dbcs, 0.)
# assemble!(asm, physics, U, :residual_and_stiffness)
# K = stiffness(asm)
# G = constraint_matrix(asm)
# # @time H = (G + I) * K
# K[asm.dof.H1_bc_dofs, asm.dof.H1_bc_dofs] .= 1.
# # R = G * residual(asm)
# # R = G * asm.residual_storage.vals
# R = asm.residual_storage
# R[asm.dof.H1_bc_dofs] .= 0.
# ΔUu = -K \ R.vals
# U.vals .= U.vals .+ ΔUu
# assemble!(asm, physics, U, :residual_and_stiffness)
# K = stiffness(asm)
# G = constraint_matrix(asm)
# # @time H = (G + I) * K
# K[asm.dof.H1_bc_dofs, asm.dof.H1_bc_dofs] .= 1.
# # R = G * residual(asm)
# # R = G * asm.residual_storage.vals
# R = asm.residual_storage
# R[asm.dof.H1_bc_dofs] .= 0.
# ΔUu = -K \ R.vals
# U.vals .= U.vals .+ ΔUu
# U
# # @time H = G * K + G * I
# # @time H = G * K
