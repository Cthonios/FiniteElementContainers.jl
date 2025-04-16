import KernelAbstractions as KA
using Adapt
using CUDA
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

struct Poisson <: AbstractPhysics{1, 0, 0}
end

function FiniteElementContainers.residual(::Poisson, cell, u_el, args...)
  (; X_q, N, ∇N_X, JxW) = cell
  ∇u_q = u_el * ∇N_X
  R_q = ∇u_q * ∇N_X' - N' * f(X_q, 0.0)
  return JxW * R_q[:]
end

function FiniteElementContainers.stiffness(::Poisson, cell, u_el, args...)
  (; X_q, N, ∇N_X, JxW) = cell
  K_q = ∇N_X * ∇N_X'
  return JxW * K_q
end

function poisson_cuda()
  # do all setup on CPU
  # the mesh for instance is not gpu compatable
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange)
  physics = Poisson()
  u = ScalarFunction(V, :u)
  asm = SparseMatrixAssembler(H1Field, u)
  pp = PostProcessor(mesh, output_file, u)

  dbcs = DirichletBC[
    DirichletBC(asm.dof, :u, :sset_1, bc_func),
    DirichletBC(asm.dof, :u, :sset_2, bc_func),
    DirichletBC(asm.dof, :u, :sset_3, bc_func),
    DirichletBC(asm.dof, :u, :sset_4, bc_func),
  ]
  # TODO this one will be tough to do on the GPU
  update_dofs!(asm, dbcs)

  # create parameters on CPU
  # TODO make a better constructor
  p = create_parameters(asm, physics, dbcs)

  # need to assemble once before moving to GPU
  # TODO try to wrap this in the |> gpu call
  U = create_field(asm, H1Field)
  @time assemble!(asm, physics, U, :stiffness)
  K = stiffness(asm)

  # device movement
  p = p |> gpu
  asm_gpu = asm |> gpu

  solver = NewtonSolver(IterativeSolver(asm_gpu, :CgSolver))
  Uu = create_unknowns(asm_gpu)
  update_bcs!(H1Field, solver, Uu, p)
  FiniteElementContainers.solve!(solver, Uu, p)

  # transfer to cpu to post-process
  p = p |> cpu
  U = p.h1_field

  write_times(pp, 1, 0.0)
  write_field(pp, 1, U)
  close(pp)
end

@time poisson_cuda()
@time poisson_cuda()
