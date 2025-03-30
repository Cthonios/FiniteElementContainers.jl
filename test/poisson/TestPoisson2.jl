# using BenchmarkTools
using Exodus
using FiniteElementContainers
using Krylov
using Parameters

# mesh file
gold_file = "./poisson/poisson.gold"
mesh_file = "./poisson/poisson.g"
output_file = "./poisson/poisson.e"

# methods for a simple Poisson problem
f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
bc_func(_, _) = 0.

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

# read mesh and relevant quantities

function poisson_v2()
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Poisson()
  u = ScalarFunction(V, :u)
  dof = DofManager(u)
  asm = SparseMatrixAssembler(dof, H1Field)
  pp = PostProcessor(mesh, output_file, u)

  # setup and update bcs
  dbcs = DirichletBC[
    DirichletBC(asm.dof, :u, :sset_1, bc_func),
    DirichletBC(asm.dof, :u, :sset_2, bc_func),
    DirichletBC(asm.dof, :u, :sset_3, bc_func),
    DirichletBC(asm.dof, :u, :sset_4, bc_func),
  ]
  update_dofs!(asm, dbcs)

  # pre-setup some scratch arrays
  Uu = create_unknowns(asm)
  Ubc = create_bcs(asm, H1Field)
  U = create_field(asm, H1Field)

  update_field!(U, asm, Uu, Ubc)
  assemble!(asm, physics, U, :residual_and_stiffness)
  K = stiffness(asm)

  for n in 1:5
    Ru = residual(asm)
    # ΔUu = -K \ Ru
    ΔUu, stat = cg(-K, Ru)
    update_field_unknowns!(U, asm.dof, ΔUu, +)
    assemble!(asm, physics, U, :residual)

    @show norm(ΔUu) norm(Ru)

    if norm(ΔUu) < 1e-12 || norm(Ru) < 1e-12
      break
    end
  end

  @show maximum(U)

  # copy_mesh(mesh_file, output_file)
  # exo = ExodusDatabase(output_file, "rw")
  # write_names(exo, NodalVariable, ["u"])
  # write_time(exo, 1, 0.0)
  # write_values(exo, NodalVariable, 1, "u", U[1, :])
  # close(exo)

  write_times(pp, 1, 0.0)
  write_field(pp, 1, U)
  close(pp)

  @test exodiff(output_file, gold_file)
  rm(output_file; force=true)
end

poisson_v2()
@time poisson_v2()
# @benchmark poisson_v2()
