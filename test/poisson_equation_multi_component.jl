# methods
f_u(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])
f_v(X, _) = 2. * π^2 * cos(π * X[1]) * cos(π * X[2])

function residual(cell, u_el)
  @unpack X, N, ∇N_X, JxW = cell
  ∇u_q = ∇N_X' * u_el[1, :]
  ∇v_q = ∇N_X' * u_el[2, :]
  R_u_q = (∇N_X * ∇u_q)' .- N' * f_u(X, 0.0)
  R_v_q = (∇N_X * ∇v_q)' .- N' * f_v(X, 0.0)
  R_q = JxW * vcat(R_u_q, R_v_q)[:]
  return R_q
end

tangent(cell, u_el) = ForwardDiff.jacobian(z -> residual(cell, z), u_el)

# script
mesh = Mesh("./meshes/mesh_test.g"; nsets=[1, 2, 3, 4])

bcs = [
  EssentialBC(mesh, 1, 1)
  EssentialBC(mesh, 2, 1)
  EssentialBC(mesh, 3, 1)
  EssentialBC(mesh, 4, 1)
  EssentialBC(mesh, 1, 2)
  EssentialBC(mesh, 2, 2)
  EssentialBC(mesh, 3, 2)
  EssentialBC(mesh, 4, 2)
]

# re        = ReferenceFE(Quad4(1), Int32, Float64)
fspace    = FunctionSpace(mesh, 1, 1)
fspaces   = [fspace]
dof       = DofManager(mesh, 2, bcs)
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

copy_mesh("./meshes/mesh_test.g", "poisson_output_multi_component.e")
exo = ExodusDatabase("poisson_output_multi_component.e", "rw")
write_names(exo, NodalVariable, ["u", "v"])
write_time(exo, 1, 0.0)
write_values(exo, NodalVariable, 1, "u", U[1, :])
write_values(exo, NodalVariable, 1, "v", U[2, :])
close(exo)

# exodiff("poisson_output_multi_component.e.gold", "poisson_output_multi_component.e")

Base.rm("poisson_output_multi_component.e")
