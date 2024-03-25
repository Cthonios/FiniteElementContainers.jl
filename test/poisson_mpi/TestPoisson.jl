using Exodus
using FiniteElementContainers
using IterativeSolvers
using LinearAlgebra
using Parameters
using PartitionedArrays
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
n_procs = 4
q_degree = 2
ranks = LinearIndices((4,))

mesh_file = "poisson.g"
global_to_color = Exodus.collect_global_to_color(mesh_file, n_procs)
# parts = uniform_partition(ranks, ExodusDatabase, mesh_file)
parts = partition_from_color(ranks, mesh_file, global_to_color)

helpers = map(ranks, parts) do rank, part
  mesh    = FileMesh(ExodusDatabase, mesh_file * ".$n_procs.$(rank - 1)")
  coords  = coordinates(mesh)
  coords  = NodalField{size(coords), Vector}(coords)
  conn    = element_connectivity(mesh, 1)
  conn    = ElementField{size(conn), Vector}(convert.(Int64, conn))
  elem    = element_type(mesh, 1) |> uppercase
  nsets   = nodeset.((mesh,), [1, 2, 3, 4])
  nsets   = map(nset -> convert.(Int64, nset), nsets)
  dof     = DofManager{1, size(coords, 2), Vector{Float64}}()
  fspaces = NonAllocatedFunctionSpace[
    NonAllocatedFunctionSpace(dof, conn, q_degree, elem)
  ]

  # now some more setup
  asm      = StaticAssembler(dof, fspaces)
  bc_nodes = sort!(unique!(vcat(nsets...)))
  update_unknown_dofs!(dof, bc_nodes)
  update_unknown_dofs!(asm, dof, fspaces, bc_nodes)

  return asm, dof, fspaces, coords
end

# setup some sparse parrays
IV = map(parts) do part
  Is = part.own.own_to_global
  vs = rand(Float64, length(globals))
  return Is, vs
end

II, VV = tuple_of_arrays(IV)
Uu = pvector(II, VV, parts) |> fetch

local_values(Uu)