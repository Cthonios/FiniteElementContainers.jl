using Exodus
using FiniteElementContainers
using LinearAlgebra
using Parameters
using Printf
using ReferenceFiniteElements
using StaticArrays

f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])

function residual(cell, u_el)
  @unpack X, N, ∇N_X, JxW = cell
  ∇u_q = ∇N_X' * u_el
  R_q = (∇N_X * ∇u_q)' .- N' * f(X, 0.0)
  return JxW * R_q[:]
end

function tangent(cell, _)
  @unpack X, N, ∇N_X, JxW = cell
  K_q = ∇N_X * ∇N_X'
  return JxW * K_q
end

function solve(mesh, fspaces, dof, assembler, bcs)
  Uu = create_unknowns(dof)
  U  = create_fields(dof)

  update_bcs!(U, bcs)
  update_fields!(U, dof, Uu)
  assemble!(assembler, mesh, fspaces, dof, residual, tangent, U)

  K = assembler.K[dof.unknown_indices, dof.unknown_indices]

  for n in 1:10
    update_bcs!(U, bcs)
    update_fields!(U, dof, Uu)
    assemble!(assembler, mesh, fspaces, dof, residual, U)

    R = assembler.R[dof.unknown_indices]

    ΔUu = -K \ R

    @printf "|R|   = %1.6e  |ΔUu| = %1.6e\n" norm(R) norm(ΔUu)

    if (norm(R) < 1e-12) || (norm(ΔUu) < 1e-12)
      break
    end
    Uu = Uu + ΔUu
  end

  update_fields!(U, dof, Uu)
  return U
end

function run_simulation()
  mesh = Mesh("mesh.g"; nsets=[1, 2, 3, 4])

  bcs = [
    EssentialBC(mesh, 1, 1)
    EssentialBC(mesh, 2, 1)
    EssentialBC(mesh, 3, 1)
    EssentialBC(mesh, 4, 1)
  ]

  q_degree = 2
  n_dof    = 1
  block_id = 1

  fspace    = FunctionSpace(mesh, block_id, q_degree)
  fspaces   = [fspace]
  dof       = DofManager(mesh, n_dof, bcs)
  assembler = StaticAssembler(dof)
  U         = solve(mesh, fspaces, dof, assembler, bcs)
  copy_mesh("mesh.g", "poisson_output.e")
  exo = ExodusDatabase("poisson_output.e", "rw")
  write_names(exo, NodalVariable, ["u"])
  write_time(exo, 1, 0.0)
  write_values(exo, NodalVariable, 1, "u", U)
  close(exo)
  
end