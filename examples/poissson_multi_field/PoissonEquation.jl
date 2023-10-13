using Exodus
using FiniteElementContainers
using ForwardDiff
using LinearAlgebra
using Parameters
using Printf
using ReferenceFiniteElements
using StaticArrays
using TimerOutputs

f_u(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
f_v(X, _) = 2. * π^2 * sin(π * X[1]) * cos(π * X[2])
f_w(X, _) = 2. * π^2 * cos(π * X[1]) * cos(π * X[2])

function residual(cell, u_el)
  @unpack X, N, ∇N_X, JxW = cell
  ∇u_q = ∇N_X' * u_el[1, :]
  ∇v_q = ∇N_X' * u_el[2, :]
  ∇w_q = ∇N_X' * u_el[3, :]
  R_u_q = (∇N_X * ∇u_q)' .- N' * f_u(X, 0.0)
  R_v_q = (∇N_X * ∇v_q)' .- N' * f_v(X, 0.0)
  R_w_q = (∇N_X * ∇w_q)' .- N' * f_w(X, 0.0)
  R_q = JxW * vcat(R_u_q, R_v_q, R_w_q)[:]
  return R_q
end

tangent(cell, u_el) = ForwardDiff.jacobian(z -> residual(cell, z), u_el)

# below works for this example but is not performant at the moment
# function tangent(cell, u_el)
#   @unpack X, N, ∇N_X, JxW = cell
#   K_uu = ∇N_X * ∇N_X'
#   K_vv = ∇N_X * ∇N_X'
#   K_ww = ∇N_X * ∇N_X'

#   NDof, N = size(u_el)
#   NxNDof = N * NDof
#   # K = Matrix{Float64}(undef, N * NDof, N * NDof)
#   # K = zeros(Float64, N * NDof, N * NDof)
#   K = zeros(MMatrix{NxNDof, NxNDof, Float64, NxNDof * NxNDof})

#   for i in 1:N
#     for j in 1:N
#       I = NDof * (i - 1) + 1
#       J = NDof * (j - 1) + 1
#       K[I, J]         = K_uu[i, j]
#       K[I + 1, J + 1] = K_vv[i, j]
#       K[I + 2, J + 2] = K_ww[i, j]
#     end
#   end

#   return JxW * K
# end

function solve(mesh, fspaces, dof, assembler, bcs, timer)
  # Uu = create_unknowns(dof)
  U  = create_fields(dof)

  @timeit timer "Boundary Conditions" begin
    @timeit timer "Update Boundary Conditions" update_bcs!(U, mesh, dof, bcs)
    # @timeit timer "Update Fields"  update_fields!(U, dof, Uu)
  end

  # U = create_fields(dof)
  Uu = create_unknowns(dof)

  @timeit timer "Assembly" begin
    @timeit timer "Assemble Stiffness and Residual" assemble!(assembler, fspaces, dof, residual, tangent, U)
  end
  K = assembler.K[dof.unknown_indices, dof.unknown_indices]
  
  for n in 1:10
    # @timeit timer "Boundary Conditions" begin
    #   # @timeit timer "Update Boundary Conditions" update_bcs!(U, mesh, dof, bcs)
    #   @timeit timer "Update Fields" update_fields!(U, dof, Uu)
    # end

    @timeit timer "Update Fields" update_fields!(U, dof, Uu)

    @timeit timer "Assembly" begin
      @timeit timer "Assemble Residual" assemble!(assembler, fspaces, dof, residual, U)
    end
    R = assembler.R[dof.unknown_indices]

    @timeit timer "Non-linear Solve" begin
      @timeit timer "Linear Solve" ΔUu = -K \ R
    end

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
  timer = TimerOutput()
  @timeit timer "Mesh" mesh = Mesh("mesh.g"; nsets=[1, 2, 3, 4])

  @timeit timer "Boundary Conditions" @timeit timer "Setup" bcs = [
    EssentialBC(mesh, 1, 1)
    EssentialBC(mesh, 2, 1)
    EssentialBC(mesh, 3, 1)
    EssentialBC(mesh, 4, 1)
    EssentialBC(mesh, 1, 2)
    EssentialBC(mesh, 2, 2)
    EssentialBC(mesh, 3, 2)
    EssentialBC(mesh, 4, 2)
    EssentialBC(mesh, 1, 3)
    EssentialBC(mesh, 2, 3)
    EssentialBC(mesh, 3, 3)
    EssentialBC(mesh, 4, 3)
  ]

  q_degree = 2
  n_dof    = 3
  block_id = 1

  @timeit timer "Function Spaces" fspace = FunctionSpace(mesh, block_id, q_degree)
  fspaces   = [fspace]
  # conn      = mesh.blocks[1].conn
  # conns     = [conn]
  @timeit timer "DofManager" dof = DofManager(mesh, n_dof)
  @timeit timer "Assembly" begin
    @timeit timer "Setup" assembler = StaticAssembler(dof)
  end
  U = solve(mesh, fspaces, dof, assembler, bcs, timer)

  @timeit timer "Post-processing" begin
    copy_mesh("mesh.g", "poisson_output.e")
    exo = ExodusDatabase("poisson_output.e", "rw")
    write_names(exo, NodalVariable, ["u", "v", "w"])
    write_time(exo, 1, 0.0)
    write_values(exo, NodalVariable, 1, "u", U[1, :])
    write_values(exo, NodalVariable, 1, "v", U[2, :])
    write_values(exo, NodalVariable, 1, "w", U[3, :])
    close(exo)
  end
  
  timer
end