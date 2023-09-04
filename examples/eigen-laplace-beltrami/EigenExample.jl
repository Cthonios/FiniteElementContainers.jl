using Exodus
using FiniteElementContainers
using IterativeSolvers
using LinearAlgebra
using Parameters
using Printf
using ReferenceFiniteElements
using StaticArrays

# really just a place holder
function residual(cell, u_el)
  @unpack ξ, N, ∇N_X, JxW = cell
  ∇u_q = ∇N_X' * u_el
  R_q = (∇N_X * ∇u_q)'
  return JxW * R_q[:]
end

function stiffness(cell, _)
  @unpack ξ, N, ∇N_X, JxW = cell
  K_q = ∇N_X * ∇N_X'
  return JxW * K_q
end

function mass(cell, _)
  @unpack ξ, N, ∇N_X, JxW = cell
  M_q = N * N'
  return JxW * M_q
end

function solve(fspace, dof, assembler, assembler_cache)
  Uu = create_unknowns(dof)
  U = create_fields(dof)

  update_bcs!(U, dof)
  update_fields!(U, dof, Uu)

  assemble!(assembler, assembler_cache,
           fspace, dof,
           residual, stiffness, mass,
           U)

  results = lobpcg(assembler.K, assembler.M, false, 100)

  return results.λ, results.X
end

function post_process(eig_vals, eig_vecs)
  copy_mesh("mesh.g", "output.e")
  exo = ExodusDatabase("output.e", "rw")
  write_names(exo, NodalVariable, ["u"])
  for (n, eig_val) in enumerate(eig_vals)
    write_time(exo, n, eig_val)
    write_values(exo, NodalVariable, n, "u", eig_vecs[:, n])
  end
  close(exo)
end

function run_simulations()
  mesh = Mesh("mesh.g", [1]; nsets=[1, 2, 3, 4])

  bcs = EssentialBC[]

  re        = ReferenceFE(Quad4(2), Int32, Float64)
  @time fspace    = FunctionSpace(mesh.coords, mesh.blocks[1], re)
  @time dof       = DofManager(mesh, 1, bcs)
  @time assembler = Assembler(dof)
  @time assembler_cache = AssemblerCache(dof, fspace)
  println("Setup complete")

  @time eig_vals, eig_vecs = solve(fspace, dof, assembler, assembler_cache)
  @time post_process(eig_vals, eig_vecs)
end