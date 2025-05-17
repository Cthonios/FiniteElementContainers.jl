import KernelAbstractions as KA
using Adapt
using AMDGPU
using Exodus
using FiniteElementContainers
using Krylov
using LinearAlgebra

# mesh file
gold_file = "./poisson/poisson.gold"
mesh_file = "./poisson/poisson.g"
output_file = "./poisson/poisson.e"

# methods for a simple Poisson problem
f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
bc_func(_, _) = 0.

include("TestPoissonCommon.jl")

function poisson_amdgpu()
  # do all setup on CPU
  # the mesh for instance is not gpu compatable
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange)
  physics = Poisson()
  u = ScalarFunction(V, :u)
  dof = DofManager(u)
  asm = SparseMatrixAssembler(dof, H1Field)

  dbcs = DirichletBC[
    DirichletBC(:u, :sset_1, bc_func),
    DirichletBC(:u, :sset_2, bc_func),
    DirichletBC(:u, :sset_3, bc_func),
    DirichletBC(:u, :sset_4, bc_func),
  ]

  # test iterative solver
  p = create_parameters(asm, physics; dirichlet_bcs=dbcs)

  # device movement
  p_gpu = p |> rocm
  asm_gpu = asm |> rocm

  solver = NewtonSolver(IterativeLinearSolver(asm_gpu, :CgSolver))
  integrator = QuasiStaticIntegrator(solver)
  @time evolve!(integrator, p_gpu)

  display(solver.timer)

  p = p_gpu |> cpu
  U = p.h1_field

  pp = PostProcessor(mesh, output_file, u)
  write_times(pp, 1, 0.0)
  write_field(pp, 1, U)
  close(pp)

  if !Sys.iswindows()
    @test exodiff(output_file, gold_file)
  end
  rm(output_file; force=true)
  display(solver.timer)
end

@time poisson_amdgpu()
@time poisson_amdgpu()

# @benchmark poisson_cuda()
