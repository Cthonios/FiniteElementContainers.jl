# using Exodus
# using FiniteElementContainers
# using LinearAlgebra
# using Parameters

f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])

function residual(cell, u_el)
  @unpack X_q, N, ∇N_X, JxW = cell
  ∇u_q = u_el * ∇N_X
  R_q = ∇u_q * ∇N_X' - N' * f(X_q, 0.0)
  return JxW * R_q[:]
end

function tangent(cell, _)
  @unpack X_q, N, ∇N_X, JxW = cell
  K_q = ∇N_X * ∇N_X'
  return JxW * K_q
end

# set up initial containers
mesh    = Mesh(ExodusDatabase, "./mesh.g")
fspaces = NonAllocatedFunctionSpace[
  NonAllocatedFunctionSpace(mesh, 1, 2)
]
dof     = DofManager(mesh, 1)
asm     = Assembler(dof, fspaces)

# set up bcs
update_unknown_ids!(dof, mesh.nset_nodes, 1)

# @show dof.is_unknown

# now pre-allocate arrays
X   = mesh.coords
U   = create_field(dof, :u)
Uu  = create_unknowns(dof)
# Uu  .= 1.0

function solve(asm, dof, fspaces, X, U, Uu)
  for n in 1:10
    update_fields!(U, dof, Uu)
    assemble!(asm, dof, fspaces, residual, tangent, X, U)
    R, K = remove_constraints(asm, dof)
    ΔUu = -K \ R
    @show norm(ΔUu) norm(R)
    if norm(R) < 1e-12
      println("Converged")
      break
    end
    Uu = Uu + ΔUu
  end
  return Uu
end

Uu = solve(asm, dof, fspaces, X, U, Uu)
update_fields!(U, dof, Uu)


copy_mesh("./mesh.g", "./output.e")
exo = ExodusDatabase("./output.e", "rw")
write_names(exo, NodalVariable, [field_names(U) |> String])
write_time(exo, 1, 0.0)
write_values(exo, NodalVariable, 1, field_names(U) |> String, U.vals[1, :])
close(exo)

@test exodiff("./output.e", "./poisson.gold")