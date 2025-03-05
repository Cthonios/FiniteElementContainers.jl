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
