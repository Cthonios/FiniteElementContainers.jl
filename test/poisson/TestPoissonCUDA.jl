import KernelAbstractions as KA
using Adapt
using CUDA
using Exodus
using FiniteElementContainers
using Krylov
using LinearAlgebra
using Test

# mesh file
gold_file = Base.source_dir() * "/poisson.gold"
mesh_file = Base.source_dir() * "/poisson.g"
output_file = Base.source_dir() * "/poisson.e"

# methods for a simple Poisson problem
f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
bc_func(_, _) = 0.

include("TestPoissonCommon.jl")

function poisson_cuda()
  # do all setup on CPU
  # the mesh for instance is not gpu compatable
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange)
  physics = Poisson()
  props = create_properties(physics)
  u = ScalarFunction(V, :u)
  asm = SparseMatrixAssembler(H1Field, u)
  pp = PostProcessor(mesh, output_file, u)

  dbcs = DirichletBC[
    DirichletBC(:u, :sset_1, bc_func),
    DirichletBC(:u, :sset_2, bc_func),
    DirichletBC(:u, :sset_3, bc_func),
    DirichletBC(:u, :sset_4, bc_func),
  ]

  # create parameters on CPU
  # TODO make a better constructor
  p = create_parameters(asm, physics, props; dirichlet_bcs=dbcs)

  # device movement
  p_gpu = p |> cuda
  asm_gpu = asm |> cuda

  solver = NewtonSolver(IterativeLinearSolver(asm_gpu, :CgSolver))
  integrator = QuasiStaticIntegrator(solver)
  evolve!(integrator, p_gpu)

  # transfer to cpu to post-process
  p = p_gpu |> cpu
  U = p.h1_field

  write_times(pp, 1, 0.0)
  write_field(pp, 1, U)
  close(pp)

  if !Sys.iswindows()
    @test exodiff(output_file, gold_file)
  end
  rm(output_file; force=true)
  display(solver.timer)
end

@time poisson_cuda()
@time poisson_cuda()
