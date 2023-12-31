# FiniteElementContainers [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://cmhamel.github.io/FiniteElementContainers.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cmhamel.github.io/FiniteElementContainers.jl/dev/) [![Build Status](https://github.com/cmhamel/FiniteElementContainers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cmhamel/FiniteElementContainers.jl/actions/workflows/CI.yml?query=branch%3Amain) [![Coverage](https://codecov.io/gh/cmhamel/FiniteElementContainers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/cmhamel/FiniteElementContainers.jl)

This package is meant to serve as a light weight and allocation free set of containers for carrying out finite element or finite element-like calculations.

The goal is to be platform independent and leverage ReferenceFiniteElements.jl for easy implementation of new containers for new element types/formulations.

Below is an example for using the package to solve the Poisson equation with a simple forcing function

```julia
using Exodus
using FiniteElementContainers
using LinearAlgebra
using Printf
using ReferenceFiniteElements
using StaticArrays
using StructArrays


# methods
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

# script
mesh = Mesh("./meshes/mesh_test.g", [1]; nsets=[1, 2, 3, 4])

bcs = [
  EssentialBC(mesh, 1, 1)
  EssentialBC(mesh, 2, 1)
  EssentialBC(mesh, 3, 1)
  EssentialBC(mesh, 4, 1)
]

re        = ReferenceFE(Quad4(1), Int32, Float64)
fspace    = FunctionSpace(mesh.coords, mesh.blocks[1], re)
dof       = DofManager(mesh, 1, bcs)
assembler = Assembler(fspace, dof)
println("Setup complete")

function solve(fspace, dof, assembler)
  Uu = create_unknowns(dof)
  Uu .= 1.0
  U = create_fields(dof)

  update_bcs!(U, dof)
  update_fields!(U, dof, Uu)

  reset!(assembler)
  update_scratch!(assembler, fspace, poisson_residual_kernel, poisson_tangent_kernel, U)
  assemble!(assembler, fspace, dof)

  K = assembler.K[dof.unknown_indices, dof.unknown_indices]

  for n in 1:10

    update_bcs!(U, dof)
    update_fields!(U, dof, Uu)

    reset!(assembler)
    update_scratch!(assembler, fspace, poisson_residual_kernel, poisson_tangent_kernel, U)
    assemble!(assembler, fspace, dof)

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

println("Solving")
U = solve(fspace, dof, assembler)

exo = ExodusDatabase("./meshes/mesh_test.g", "r")
Exodus.copy(exo, "output.e")
close(exo)
exo = ExodusDatabase("poisson_output.e", "rw")
write_number_of_variables(exo, NodalVariable, 1)
write_names(exo, NodalVariable, ["u"])
write_time(exo, 1, 0.0)
write_values(exo, NodalVariable, 1, "u", U)
close(exo)

@exodiff "poisson_output.e.gold" "poisson_output.e"

Base.rm("poisson_output.e")
```