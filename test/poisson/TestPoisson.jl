using Exodus
using FiniteElementContainers
using IterativeSolvers
using LinearAlgebra
using Parameters
using SparseArrays

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

types = [Matrix, Vector]
# type = Vector

for type in types

  mesh_new = FileMesh(ExodusDatabase, "./poisson/poisson.g")
  coords  = coordinates(mesh_new)
  coords  = NodalField{size(coords), type}(coords)
  conn    = element_connectivity(mesh_new, 1)
  conn    = ElementField{size(conn), type}(convert.(Int64, conn))
  elem    = FiniteElementContainers.element_type(mesh_new, 1)
  nsets   = nodeset.((mesh_new,), [1, 2, 3, 4])
  nsets   = map(nset -> convert.(Int64, nset), nsets)
  dof     = DofManager{1, size(coords, 2), type{Float64}}()
  fspaces = NonAllocatedFunctionSpace[
    NonAllocatedFunctionSpace(dof, conn, 2, elem)
  ]
  asm     = StaticAssembler(dof, fspaces)

  # set up bcs
  # update_unknown_ids!(dof, nsets, [1, 1, 1, 1])
  bc_nodes = sort!(unique!(vcat(nsets...)))
  update_unknown_dofs!(dof, bc_nodes)
  update_unknown_dofs!(asm, dof, fspaces, bc_nodes)


  # now pre-allocate arrays
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


  copy_mesh("./poisson/poisson.g", "./poisson/poisson_$type.e")
  exo = ExodusDatabase("./poisson/poisson_$type.e", "rw")
  write_names(exo, NodalVariable, ["u"])
  write_time(exo, 1, 0.0)
  if type == Vector
    write_values(exo, NodalVariable, 1, "u", U.vals)
  elseif type == Matrix
    write_values(exo, NodalVariable, 1, "u", U.vals[1, :])
  end
  close(exo)
  @test exodiff("./poisson/poisson_$type.e", "./poisson/poisson.gold")
  rm("./poisson/poisson_$type.e"; force=true)
end