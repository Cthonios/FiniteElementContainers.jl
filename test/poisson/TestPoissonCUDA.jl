import KernelAbstractions as KA
using Adapt
using BenchmarkTools
using CUDA
using Exodus
using FiniteElementContainers
using Krylov
using LinearAlgebra
using Parameters
using ReferenceFiniteElements
using SparseArrays

# methods for a simple Poisson problem
f(X, _) = 2. * π^2 * sin(2π * X[1]) * sin(2π * X[2])
# f(X, _) = 0.
bc_func(_, _) = 0.
bc_func_2(_, _) = 0.

struct Poisson <: AbstractPhysics{0, 0}
end

function FiniteElementContainers.residual(::Poisson, cell, u_el, args...)
  @unpack X_q, N, ∇N_X, JxW = cell
  ∇u_q = u_el * ∇N_X
  R_q = ∇u_q * ∇N_X' - N' * f(X_q, 0.0)
  return JxW * R_q[:]
end

function FiniteElementContainers.stiffness(::Poisson, cell, u_el, args...)
  @unpack X_q, N, ∇N_X, JxW = cell
  K_q = ∇N_X * ∇N_X'
  return JxW * K_q
end

function poisson_cuda()
  # do all setup on CPU
  # the mesh for instance is not gpu compatable
  mesh = UnstructuredMesh("./test/poisson/poisson.g")
  V = FunctionSpace(mesh, H1Field, Lagrange)
  physics = Poisson()
  u = ScalarFunction(V, :u)
  dof = DofManager(u)
  asm = SparseMatrixAssembler(dof, H1Field)

  dbcs = DirichletBC[
    DirichletBC(asm.dof, :u, :sset_1, bc_func),
    DirichletBC(asm.dof, :u, :sset_2, bc_func),
    DirichletBC(asm.dof, :u, :sset_3, bc_func_2),
    DirichletBC(asm.dof, :u, :sset_4, bc_func_2),
  ]
  # TODO this one will be tough to do on the GPU
  update_dofs!(asm, dbcs)

  # need to assemble once before moving to GPU
  # TODO try to wrap this in the |> gpu call
  U = create_field(asm, H1Field)
  assemble!(asm, physics, U, :stiffness)
  K = stiffness(asm)

  # device movement
  asm_gpu = asm |> gpu
  dbcs_gpu = dbcs .|> gpu

  Uu = create_unknowns(asm_gpu)
  Ubc = create_bcs(asm_gpu, H1Field)
  # Ubc = create_bcs(asm_gpu.dof, H1Field, dbcs_gpu, 0.0)
  # Ubc = create_bcs(asm.dof, H1Field, dbcs, 0.0) |> gpu
  U = create_field(asm_gpu, H1Field)

  update_field!(U, asm_gpu, Uu, Ubc)
  assemble!(asm_gpu, physics, U, :residual)
  assemble!(asm_gpu, physics, U, :stiffness)

  # Ru = residual(asm_gpu)
  K = stiffness(asm_gpu)

  for n in 1:3
    # ΔUu = -K \ Ru
    Ru = residual(asm_gpu)
    ΔUu, stats = cg(-K, Ru)
    update_field_unknowns!(U, asm_gpu.dof, ΔUu, +)
    assemble!(asm_gpu, physics, U, :residual)

    @show norm(ΔUu) norm(Ru)

    if norm(ΔUu) < 1e-12 || norm(Ru) < 1e-12
      break
    end
  end

  # update_field!(U, asm_gpu, Uu, Ubc)
  U = U |> cpu

  copy_mesh("./test/poisson/poisson.g", "./test/poisson/poisson.e")
  exo = ExodusDatabase("./test/poisson/poisson.e", "rw")
  write_names(exo, NodalVariable, ["u"])
  write_time(exo, 1, 0.0)
  write_values(exo, NodalVariable, 1, "u", U[1, :])
  close(exo)
end

@time poisson_cuda()
@time poisson_cuda()

# @benchmark poisson_cuda()
