using Exodus
using FiniteElementContainers
using StaticArrays
using Test

# mesh file
gold_file = Base.source_dir() * "/poisson.gold"
mesh_file = Base.source_dir() * "/poisson.g"
output_file = Base.source_dir() * "/poisson.e"

# methods for a simple Poisson problem
f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
# f(_, _) = 1.
bc_func(_, _) = 0.
bc_func_neumann(_, _) = SVector{1, Float64}(1.)

include("TestPoissonCommon.jl")

# read mesh and relevant quantities

function test_poisson_direct()
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Poisson()
  props = create_properties(physics)
  u = ScalarFunction(V, :u)
  asm = SparseMatrixAssembler(H1Field, u)

  # setup and update bcs
  dbcs = DirichletBC[
    DirichletBC(:u, :sset_1, bc_func),
    DirichletBC(:u, :sset_2, bc_func),
    DirichletBC(:u, :sset_3, bc_func),
    DirichletBC(:u, :sset_4, bc_func),
  ]

  # direct solver test
  # setup the parameters
  @show p = create_parameters(asm, physics, props; dirichlet_bcs=dbcs)

  # setup solver and integrator
  solver = NewtonSolver(DirectLinearSolver(asm))
  integrator = QuasiStaticIntegrator(solver)
  evolve!(integrator, p)

  pp = PostProcessor(mesh, output_file, u)
  write_times(pp, 1, 0.0)
  write_field(pp, 1, p.h1_field)
  close(pp)

  if !Sys.iswindows()
    @test exodiff(output_file, gold_file)
  end
  rm(output_file; force=true)
  display(solver.timer)

end

function test_poisson_direct_neumman()
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Poisson()
  props = create_properties(physics)
  u = ScalarFunction(V, :u)
  asm = SparseMatrixAssembler(H1Field, u)

  # setup and update bcs
  dbcs = DirichletBC[
    DirichletBC(:u, :sset_1, bc_func),
    DirichletBC(:u, :sset_2, bc_func)
  ]

  nbcs = NeumannBC[
    NeumannBC(:u, :sset_3, bc_func_neumann),
    NeumannBC(:u, :sset_4, bc_func_neumann)
  ]

  # direct solver test
  # setup the parameters
  @show p = create_parameters(
    asm, physics, props; 
    dirichlet_bcs=dbcs,
    neumann_bcs=nbcs
  )

  # setup solver and integrator
  solver = NewtonSolver(DirectLinearSolver(asm))
  integrator = QuasiStaticIntegrator(solver)
  evolve!(integrator, p)

  pp = PostProcessor(mesh, output_file, u)
  write_times(pp, 1, 0.0)
  write_field(pp, 1, p.h1_field)
  close(pp)

  if !Sys.iswindows()
    @test exodiff(output_file, gold_file)
  end
  rm(output_file; force=true)
  display(solver.timer)

end

function test_poisson_iterative()
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Poisson()
  props = create_properties(physics)
  u = ScalarFunction(V, :u)
  asm = SparseMatrixAssembler(H1Field, u)

  # setup and update bcs
  dbcs = DirichletBC[
    DirichletBC(:u, :sset_1, bc_func),
    DirichletBC(:u, :sset_2, bc_func),
    DirichletBC(:u, :sset_3, bc_func),
    DirichletBC(:u, :sset_4, bc_func),
  ]

  # iterative solver test
  # setup the parameters
  p = create_parameters(asm, physics, props; dirichlet_bcs=dbcs)

  # setup solver and integrator
  solver = NewtonSolver(IterativeLinearSolver(asm, :CgSolver))
  integrator = QuasiStaticIntegrator(solver)
  @time evolve!(integrator, p)

  display(solver.timer)

  pp = PostProcessor(mesh, output_file, u)
  write_times(pp, 1, 0.0)
  write_field(pp, 1, p.h1_field)
  close(pp)

  if !Sys.iswindows()
    @test exodiff(output_file, gold_file)
  end
  rm(output_file; force=true)
  display(solver.timer)
end

@time test_poisson_direct()
@time test_poisson_direct()
# @time test_poisson_direct_neumman()
# @time test_poisson_direct_neumman()
@time test_poisson_iterative()
@time test_poisson_iterative()

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
