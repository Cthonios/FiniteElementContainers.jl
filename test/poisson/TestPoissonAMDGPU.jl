import KernelAbstractions as KA
using Adapt
using AMDGPU
using Exodus
using FiniteElementContainers
using Krylov
using LinearAlgebra

# mesh file
gold_file = "./test/poisson/poisson.gold"
mesh_file = "./test/poisson/poisson.g"
output_file = "./test/poisson/poisson.e"

# methods for a simple Poisson problem
f(X, _) = 2. * π^2 * sin(2π * X[1]) * sin(2π * X[2])
bc_func(_, _) = 0.

include("TestPoissonCommon.jl")

# function poisson_amdgpu()
  # do all setup on CPU
  # the mesh for instance is not gpu compatable
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange)
  physics = Poisson()
  u = ScalarFunction(V, :u)
  dof = DofManager(u)
  asm = SparseMatrixAssembler(dof, H1Field)
  pp = PostProcessor(mesh, output_file, u)

  dbcs = DirichletBC[
    DirichletBC(:u, :sset_1, bc_func),
    DirichletBC(:u, :sset_2, bc_func),
    DirichletBC(:u, :sset_3, bc_func),
    DirichletBC(:u, :sset_4, bc_func),
  ]

  p = create_parameters(asm, physics; dirichlet_bcs=dbcs)

  # device movement
  p_gpu = p |> gpu
  asm_gpu = asm |> gpu

  solver = NewtonSolver(IterativeLinearSolver(asm_gpu, :CgSolver))
  integrator = QuasiStaticIntegrator(solver)
  @time evolve!(integrator, p_gpu)

  display(solver.timer)

  p = p_gpu |> cpu
  U = p.h1_field

  write_times(pp, 1, 0.0)
  write_field(pp, 1, U)
  close(pp)

# end

# @time poisson_amdgpu()
# @time poisson_amdgpu()

# @benchmark poisson_cuda()
