using Exodus
using FiniteElementContainers
using Parameters

# methods for a simple Poisson problem
f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])

bc_func(_, _) = 0.

function residual(cell, u_el)
  @unpack X_q, N, ∇N_X, JxW = cell
  ∇u_q = u_el * ∇N_X
  R_q = ∇u_q * ∇N_X' - N' * f(X_q, 0.0)
  return JxW * R_q[:]
end

function tangent(cell, u_el)
  @unpack X_q, N, ∇N_X, JxW = cell
  K_q = ∇N_X * ∇N_X'
  return JxW * K_q
end

# read mesh and relevant quantities

function poisson_v2()
  mesh = UnstructuredMesh("./poisson/poisson.g")
  V = FunctionSpace(mesh, H1, Lagrange) 
  u = ScalarFunction(V, :u)
  asm = SparseMatrixAssembler(u)

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
  Ubc = zeros(length(asm.dof.H1_bc_dofs))
  U = create_field(asm, H1)
  R = create_field(asm, H1)

  update_field!(U, asm, Uu, Ubc)

  assemble!(R, asm, residual, U)
  assemble!(asm, tangent, U)

  K = SparseArrays.sparse!(asm)

  for n in 1:3
    ΔUu = -K \ R[asm.dof.H1_unknown_dofs]
    U[asm.dof.H1_unknown_dofs] .=+ ΔUu

    @time assemble!(R, asm, residual, U)

    if norm(ΔUu) < 1e-12 || norm(R[asm.dof.H1_unknown_dofs]) < 1e-12
      break
    end
  end

  copy_mesh("./poisson/poisson.g", "./poisson/poisson.e")
  exo = ExodusDatabase("./poisson/poisson.e", "rw")
  write_names(exo, NodalVariable, ["u"])
  write_time(exo, 1, 0.0)
  write_values(exo, NodalVariable, 1, "u", U[1, :])
  close(exo)
  @test exodiff("./poisson/poisson.e", "./poisson/poisson.gold")
  rm("./poisson/poisson.e"; force=true)
end

poisson_v2()
poisson_v2()