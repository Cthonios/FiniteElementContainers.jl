using Aqua
using Exodus
using FiniteElementContainers
using ReferenceFiniteElements
using Test
using TestSetExtensions


@testset ExtendedTestSet "DofManager" begin
  Nx, Ny = 4, 5
  mesh = Mesh(Nx, Ny, [0., 1.], [0., 1.])
  re = ReferenceFE(Tri3(2))
  fs = FunctionSpace(mesh.coords, mesh.blocks[1], re)
  bcs = [
    EssentialBC(1, 1)
    EssentialBC(3, 2)
  ]
  dof_manager = DofManager(mesh, [fs], bcs, 2)

  @test get_bc_size(dof_manager) == Nx + Ny # -1 for repeat node
  @test get_unknown_size(dof_manager) == 2 * Nx * Ny - Nx - Ny
  U = zeros(Float64, 2, Nx * Ny)
  # U[2, :]
  U[2, :]                   .= 1.0
  U[1, mesh.nsets[1].nodes] .= 2.0
  U[2, mesh.nsets[3].nodes] .= 3.0

  Uu = get_unknown_values(dof_manager, U)
  bc = get_bc_values(dof_manager, U)

  @test all(x -> x ≈ 0.0 || x ≈ 1.0, Uu)
  @test all(x -> x ≈ 2.0 || x ≈ 3.0, bc)
end


@testset ExtendedTestSet "EssentialBC" begin
  bc = EssentialBC(1, 2)
  @test bc.nset_id == 1
  @test bc.dof     == 2
end

@testset ExtendedTestSet "FunctionSpace" begin
  exo = ExodusDatabase("meshes/mesh_test.e", "r")
  coords = read_coordinates(exo)
  block = Block(exo, 1)
  re = ReferenceFE(Quad4(2))
  fs = FunctionSpace(coords, block, re)
  @test sum(fs.interpolants.JxW) ≈ 1.0
  close(exo)
end

@testset ExtendedTestSet "Mesh" begin
  exo = ExodusDatabase("meshes/mesh_test.e", "r")
  mesh = Mesh(exo, [1], [1, 2, 3, 4], [1, 2, 3, 4])
  @test read_coordinates(exo) == mesh.coords
  # @test Block(exo, 1)         == mesh.blocks[1]
  # for n in 1:4
  #   @test NodeSet(exo, n) == mesh.nsets[n]
  #   @test SideSet(exo, n) == mesh.ssets[n]
  # end
  close(exo)
end
