using Exodus
using FiniteElementContainers
using IterativeSolvers
using LinearAlgebra
using Parameters
using SparseArrays

f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])

function eigen_residual(cell, u_el)
  @unpack X_q, N, ∇N_X, JxW = cell
  ∇u_q = u_el * ∇N_X
  R_q = ∇u_q * ∇N_X' - N' * f(X_q, 0.0)
  return JxW * R_q[:]
end

function eigen_tangent(cell, _)
  @unpack X_q, N, ∇N_X, JxW = cell
  K_q = ∇N_X * ∇N_X'
  return JxW * K_q
end

function eigen_mass(cell, _)
  @unpack X_q, N, ∇N_X, JxW = cell
  M_q = N * N'
  return JxW * M_q
end

# set up initial containers

types = [Matrix, Vector]
# type = Vector

for type in types

  mesh_new = FileMesh(ExodusDatabase, "./eigen/eigen.g")
  coords  = coordinates(mesh_new)
  coords  = NodalField{size(coords), type}(coords)
  conn    = element_connectivity(mesh_new, 1)
  conn    = ElementField{size(conn), type}(convert.(Int64, conn))
  elem    = element_type(mesh_new, 1)
  nsets   = nodeset.((mesh_new,), [1, 2, 3, 4])
  nsets   = map(nset -> convert.(Int64, nset), nsets)
  dof     = DofManager{1, size(coords, 2), type{Float64}}()
  fspaces = NonAllocatedFunctionSpace[
    NonAllocatedFunctionSpace(dof, conn, 2, elem)
  ]
  asm     = DynamicAssembler(dof, fspaces)

  # set up bcs
  # update_unknown_ids!(dof, nsets, [1, 1, 1, 1])
  # bc_nodes = sort!(unique!(vcat(nsets...)))
  bc_nodes = Int64[]
  update_unknown_dofs!(dof, bc_nodes)
  update_unknown_dofs!(asm, fspaces, bc_nodes)


  # now pre-allocate arrays
  U   = create_fields(dof)
  Uu  = create_unknowns(dof)
  ΔUu = create_unknowns(dof)
  Uu  .= 1.0

  function solve(asm, dof, fspaces, X, U, Uu)
    @info "Solving"
    update_fields!(U, dof, Uu)
    assemble!(asm, dof, fspaces, X, U, eigen_residual, eigen_tangent, eigen_mass)
    K, M = sparse(asm)
    # trying a small shift manually since I can't get a cholesky preconditioner to work
    σ = 1e-5
    K = K + σ * I
    results = lobpcg(K, M, false, 100)
    return results.λ, results.X
  end

  λs, Uus = solve(asm, dof, fspaces, coords, U, Uu)

  copy_mesh("./eigen/eigen.g", "./eigen/eigen_$type.e")
  exo = ExodusDatabase("./eigen/eigen_$type.e", "rw")
  write_names(exo, NodalVariable, ["u"])

  for n in axes(λs, 1)
    update_fields!(U, dof, Uus[:, n])
    write_time(exo, n, λs[n])
    if type == Vector
      write_values(exo, NodalVariable, n, "u", U.vals)
    elseif type == Matrix
      write_values(exo, NodalVariable, n, "u", U.vals[1, :])
    end
  end
  close(exo)
  # TODO fix exodiff somehow...
  # TODO what's really going on is eigenvectors aren't necessarily ordered
  # TODO in the same way from solve to solve
  # @test exodiff("./eigen/eigen_$type.e", "./eigen/eigen.gold")
  rm("./eigen/eigen_$type.e"; force=true)
end