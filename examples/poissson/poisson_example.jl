using Exodus
using FiniteElementContainers
using LinearAlgebra
using Parameters
using Printf
using ReferenceFiniteElements
using StaticArrays

f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])

function residual(cell, u_el)
  @unpack ξ, N, ∇N_X, JxW = cell
  ∇u_q = ∇N_X' * u_el
  R_q = (∇N_X * ∇u_q)' .- N' * f(ξ, 0.0)
  return JxW * R_q[:]
end

function tangent(cell, _)
  @unpack ξ, N, ∇N_X, JxW = cell
  K_q = ∇N_X * ∇N_X'
  return JxW * K_q
end

function solve(fspace, dof, assembler, assembler_cache)
  Uu = create_unknowns(dof)
  U  = create_fields(dof)

  update_bcs!(U, dof)
  update_fields!(U, dof, Uu)
  assemble!(assembler, assembler_cache,
            fspace, dof, 
            residual, tangent, 
            U)
  
  K = assembler.K[dof.unknown_indices, dof.unknown_indices]

  for n in 1:10

    update_bcs!(U, dof)
    update_fields!(U, dof, Uu)
    assemble!(assembler, assembler_cache, fspace, dof, residual, U)

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
  mesh = Mesh("mesh.g", [1]; nsets=[1, 2, 3, 4])

  bcs = [
    EssentialBC(mesh, 1, 1)
    EssentialBC(mesh, 2, 1)
    EssentialBC(mesh, 3, 1)
    EssentialBC(mesh, 4, 1)
  ]

  re        = ReferenceFE(Quad4(1), Int32, Float64)
  fspace    = FunctionSpace(mesh.coords, mesh.blocks[1], re)
  dof       = DofManager(mesh, 1, bcs)
  assembler = Assembler(dof)
  assembler_cache = AssemblerCache(dof, fspace)
  println("Setup complete")

  @time U = solve(fspace, dof, assembler, assembler_cache)

  copy_mesh("mesh.g", "poisson_output.e")
  exo = ExodusDatabase("poisson_output.e", "rw")
  write_names(exo, NodalVariable, ["u"])
  write_time(exo, 1, 0.0)
  write_values(exo, NodalVariable, 1, "u", U)
  close(exo)
end