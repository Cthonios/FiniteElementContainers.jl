using Exodus
using FiniteElementContainers
using LinearAlgebra
using Parameters
using ReferenceFiniteElements
using SparseArrays

# methods for a simple Poisson problem
f(X, _) = 2. * π^2 * sin(2π * X[1]) * sin(4π * X[2])

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

# function poisson_v2()
  mesh = UnstructuredMesh("./test/poisson/poisson.g")
  V = FunctionSpace(mesh, H1, Lagrange) 
  u = ScalarFunction(V, :u)
  dof = NewDofManager(u)
  asm = SparseMatrixAssembler(dof)

  # setup and update bcs
  bc_nodes = sort!(unique!(vcat(values(mesh.nodeset_nodes)...)))

  Uu = create_unknowns(dof)
  Ubc = zeros(length(bc_nodes))
  U = create_field(dof, H1)
  R = create_field(dof, H1)

  @time update_dofs!(asm, bc_nodes)
  # @time update_field_bcs!(U, dof, Ubc)
  @time update_field!(U, dof, Uu, Ubc)
  # @time update_dofs!(asm, bc_nodes)

  @time assemble!(R, asm, residual, U)
  assemble!(asm, tangent, U)

  K = SparseArrays.sparse!(asm)

  for n in 1:3
    ΔUu = -K \ R[asm.dof.H1_unknown_dofs]
    U[asm.dof.H1_unknown_dofs] .=+ ΔUu
    # @time FiniteElementContainers.update_field_unknowns!(U, dof, )

    assemble!(R, asm, residual, U)

    if norm(ΔUu) < 1e-12 || norm(R[asm.dof.H1_unknown_dofs]) < 1e-12
      break
    end
  end

  copy_mesh("./test/poisson/poisson.g", "./test/poisson/poisson.e")
  exo = ExodusDatabase("./test/poisson/poisson.e", "rw")
  write_names(exo, NodalVariable, ["u"])
  write_time(exo, 1, 0.0)
  write_values(exo, NodalVariable, 1, "u", U[1, :])
  close(exo)
  # @test exodiff("./test/poisson/poisson.e", "./test/poisson/poisson.gold")
  # rm("./test/poisson/poisson_$type.e"; force=true)
# end

# poisson_v2()
# poisson_v2()