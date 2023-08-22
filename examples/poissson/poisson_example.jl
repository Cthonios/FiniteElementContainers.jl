using Exodus
using FiniteElementContainers
# using IterativeSolvers
using LinearAlgebra
# using Preconditioners
using Printf
using ReferenceFiniteElements
using StaticArrays

f(X::SVector{2, Float64}, ::Float64) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])

function poisson_residual_kernel(
  interp::FunctionSpaceInterpolant{N, D, Rtype, L},
  u_el::SMatrix{N, 1, Rtype, N}
) where {N, D, Rtype, L}

  ∇u_q = interp.∇N_X' * u_el
  R_q = (interp.∇N_X * ∇u_q)' .- interp.N' * f(interp.ξ, 0.0)
  return interp.JxW * R_q[:]
end

function poisson_tangent_kernel(
  interp::FunctionSpaceInterpolant{N, D, Rtype, L},
  ::SMatrix{N, 1, Rtype, N}
) where {N, D, Rtype, L}

  K_q = interp.∇N_X * interp.∇N_X'
  return interp.JxW * K_q
end

function solve(fspace, dof, assembler, assembler_cache)
  Uu = create_unknowns(dof)
  Uu .= 1.0
  U = create_fields(dof)
  ΔUu = create_unknowns(dof)

  update_bcs!(U, dof)
  update_fields!(U, dof, Uu)
  assemble!(assembler, assembler_cache,
            fspace, dof, 
            poisson_residual_kernel, poisson_tangent_kernel, 
            U)
  
  K = assembler.K[dof.unknown_indices, dof.unknown_indices]
  # P = DiagonalPreconditioner(K)
  # P = CholeskyPreconditioner(K, 2)

  for n in 1:10

    update_bcs!(U, dof)
    update_fields!(U, dof, Uu)
    assemble!(assembler, assembler_cache, fspace, dof, poisson_residual_kernel, U)

    R = assembler.R[dof.unknown_indices]

    ΔUu = -K \ R
    # cg!(ΔUu, -K, R, Pl=P)
    # @time gmres!(ΔUu, -K, R, Pl=P)
    # gmres!(\)

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
  @time fspace    = FunctionSpace(mesh.coords, mesh.blocks[1], re)
  @time dof       = DofManager(mesh, 1, bcs)
  @time assembler = Assembler(dof)
  @time assembler_cache = AssemblerCache(dof, fspace)
  println("Setup complete")

  @time U = solve(fspace, dof, assembler, assembler_cache)

  exo = ExodusDatabase("mesh.g", "r")
  Exodus.copy(exo, "poisson_output.e")
  close(exo)
  exo = ExodusDatabase("poisson_output.e", "rw")
  write_number_of_variables(exo, NodalVariable, 1)
  write_names(exo, NodalVariable, ["u"])
  write_time(exo, 1, 0.0)
  write_values(exo, NodalVariable, 1, "u", U)
  close(exo)
end