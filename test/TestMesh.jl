function test_structured_mesh()
  coords, conns = FiniteElementContainers.create_structured_mesh_data(3, 3, [0., 1.], [0., 1.])
  @test coords ≈ [
    0.0  0.5  1.0  0.0  0.5  1.0  0.0  0.5  1.0;
    0.0  0.0  0.0  0.5  0.5  0.5  1.0  1.0  1.0
  ]
  @test conns == [
    1  1  4  4  2  2  5  5;
    2  5  5  8  3  6  6  9;
    5  4  8  7  6  5  9  8
  ]
end

@testset ExtendedTestSet "Mesh" begin
  test_structured_mesh()
end

struct DummyMesh <: FiniteElementContainers.AbstractMesh
end

# @test ExtendedTestSet "Mesh definition" begin
  mesh = DummyMesh()
  @test_throws AssertionError coordinates(mesh)
  @test_throws AssertionError element_block_id_map(mesh, 1)
  @test_throws AssertionError element_block_ids(mesh)
  @test_throws AssertionError element_block_names(mesh)
  @test_throws AssertionError element_connectivity(mesh, 1)
  @test_throws AssertionError element_type(mesh, 1)
  @test_throws AssertionError nodeset(mesh, 1)
  @test_throws AssertionError nodesets(mesh, [1])
  @test_throws AssertionError nodeset_ids(mesh)
  @test_throws AssertionError num_dimensions(mesh)
  @test_throws AssertionError num_nodes(mesh)
  @test_throws AssertionError sideset(mesh, 1)
  @test_throws AssertionError sideset_ids(mesh)
  @test_throws AssertionError sideset_names(mesh)
# end