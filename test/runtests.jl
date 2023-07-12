using Aqua
using Exodus
using FiniteElementContainers
using ReferenceFiniteElements
using Test
using TestSetExtensions

@testset ExtendedTestSet "FunctionSpace" begin
  exo = ExodusDatabase("meshes/mesh_test.e", "r")
  coords = read_coordinates(exo)
  block = Block(exo, 1)
  re = ReferenceFE(Quad4(), 2)
  fs = FunctionSpace(coords, block, re)
  @test sum(fs.JxWs) ≈ 1.0
  close(exo)
end
