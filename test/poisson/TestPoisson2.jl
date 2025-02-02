using Exodus
using FiniteElementContainers

# methods for a simple Poisson problem
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

# read mesh and relevant quantities
mesh = UnstructuredMesh(ExodusDatabase, "./poisson/poisson.g")

# setup FEM fields
coords = NodalField(mesh.nodal_coords)
conn = ElementField(mesh.element_conns.block_1)

# setup DofManager
dof = DofManager{1, size(coords, 2), Vector{Float64}}()

# setup fspace
fspaces = NonAllocatedFunctionSpace[
  NonAllocatedFunctionSpace(dof, 1:size(conn, 2) |> collect, conn, 2, mesh.element_types.block_1)
]

# setup assembler
asm = StaticAssembler(dof, fspaces)

# setup and update bcs
bc_nodes = sort!(unique!(vcat(values(mesh.nodeset_nodes)...)))
update_unknown_dofs!(dof, bc_nodes)
update_unknown_dofs!(asm, dof, fspaces, bc_nodes)

# pre-allocate some arrays
U   = create_fields(dof)
R   = create_fields(dof)
Uu  = create_unknowns(dof)
ΔUu = create_unknowns(dof)
Uu  .= 1.0

function solve(asm, dof, fspaces, X, U, Uu)
  for n in 1:10
    update_fields!(U, dof, Uu)
    # assemble!(R, asm, dof, fspaces, X, U, residual, tangent)
    assemble!(asm, dof, fspaces, X, U, residual, tangent)
    # R = asm.residuals[dof.unknown_dofs]
    R_view = @views asm.residuals[dof.unknown_dofs]
    K = sparse(asm)
    cg!(ΔUu, -K, R_view)
    @show norm(ΔUu) norm(R_view)
    if norm(R_view) < 1e-12
      println("Converged")
      break
    end
    Uu = Uu + ΔUu
  end
  return Uu
end

Uu = solve(asm, dof, fspaces, coords, U, Uu)
update_fields!(U, dof, Uu)
