using Aqua
using Exodus
using FiniteElementContainers
using ReferenceFiniteElements
using Test
using TestSetExtensions


@testset ExtendedTestSet "EssentialBC" begin
  bc = EssentialBC(1, 2)
  @test bc.nset_id == 1
  @test bc.dof     == 2
end

@testset ExtendedTestSet "FunctionSpace" begin
  exo = ExodusDatabase("meshes/mesh_test.e", "r")
  coords = read_coordinates(exo)
  block = Block(exo, 1)
  re = ReferenceFE(Quad4(), 2)
  fs = FunctionSpace(coords, block, re)
  @test sum(fs.JxWs) ≈ 1.0
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
