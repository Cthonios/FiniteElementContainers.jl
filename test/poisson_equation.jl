# methods
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

# script
mesh = Mesh("./meshes/mesh_test.g"; nsets=[1, 2, 3, 4])

bcs = [
  EssentialBC(mesh, 1, 1)
  EssentialBC(mesh, 2, 1)
  EssentialBC(mesh, 3, 1)
  EssentialBC(mesh, 4, 1)
]

# re        = ReferenceFE(Quad4(1), Int32, Float64)
fspace    = FunctionSpace(mesh, 1, 1)
fspaces   = [fspace]
dof       = DofManager(mesh, 1, bcs)
assembler = StaticAssembler(dof)

function solve(mesh, fspaces, dof, assembler)
  Uu = create_unknowns(dof)
  U  = create_fields(dof)

  update_bcs!(U, mesh, bcs)
  update_fields!(U, dof, Uu)
  assemble!(assembler, mesh, fspaces, dof, residual, tangent, U)
  
  K = assembler.K[dof.unknown_indices, dof.unknown_indices]

  for n in 1:10
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

U = solve(mesh, fspaces, dof, assembler)

copy_mesh("./meshes/mesh_test.g", "poisson_output.e")
exo = ExodusDatabase("poisson_output.e", "rw")
write_names(exo, NodalVariable, ["u"])
write_time(exo, 1, 0.0)
write_values(exo, NodalVariable, 1, "u", U)
close(exo)

# exodiff("poisson_output.e.gold", "poisson_output.e")

Base.rm("poisson_output.e")
