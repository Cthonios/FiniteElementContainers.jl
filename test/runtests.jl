using Aqua
using Exodus
using FiniteElementContainers
using ForwardDiff
using JET
using LinearAlgebra
using Parameters
using Printf
using Test
using TestSetExtensions

@testset ExtendedTestSet "DofManager" begin
  Nx, Ny = 4, 5
  mesh = Mesh(Nx, Ny, [0., 1.], [0., 1.])
  fs = FunctionSpace(mesh, 1, 2)
  bcs = [
    EssentialBC(mesh, 1, 1)
    EssentialBC(mesh, 3, 2)
  ]
  dof_manager = DofManager(mesh, 2, bcs)

  @test dof_manager.unknown_indices |> length == 2 * Nx * Ny - Nx - Ny
  U = zeros(Float64, 2, Nx * Ny)
  U[2, :]                   .= 1.0
  U[1, mesh.nsets[1].nodes] .= 2.0
  U[2, mesh.nsets[3].nodes] .= 3.0

  Uu = U[dof_manager.is_unknown]
  bc = U[.~dof_manager.is_unknown]
  @test all(x -> x ≈ 0.0 || x ≈ 1.0, Uu)
  @test all(x -> x ≈ 2.0 || x ≈ 3.0, bc)
end


@testset ExtendedTestSet "EssentialBC" begin
  mesh = Mesh("meshes/mesh_test.g"; nsets=[1])
  bc = EssentialBC(mesh, 1, 2)
  @test bc.dof   == 2
  @test bc.nodes == mesh.nsets[1].nodes
  # @test bc.coords == mesh.coords[:, mesh.nsets[1].nodes]
  # for n in axes(bc.nodes)
  #   @test bc.coords[n][1] == mesh.coords[1, mesh.nsets[1].nodes[n]]
  #   @test bc.coords[n][2] == mesh.coords[2, mesh.nsets[1].nodes[n]]
  # end
end

@testset ExtendedTestSet "FunctionSpace" begin
  mesh = Mesh("meshes/mesh_test.g")
  fs = FunctionSpace(mesh, 1, 2)
  @test sum(fs.JxW) ≈ 1.0
end

include("poisson_equation.jl")
include("poisson_equation_multi_component.jl")

# Aqua testing
# Aqua.test_all(FiniteElementContainers)
Aqua.test_ambiguities(FiniteElementContainers)
Aqua.test_unbound_args(FiniteElementContainers)
Aqua.test_undefined_exports(FiniteElementContainers)
Aqua.test_piracy(FiniteElementContainers)
Aqua.test_project_extras(FiniteElementContainers)
Aqua.test_stale_deps(FiniteElementContainers)
Aqua.test_deps_compat(FiniteElementContainers)
Aqua.test_project_toml_formatting(FiniteElementContainers)

# JET testing
# test_package("FiniteElementContainers"; target_defined_modules=true)
