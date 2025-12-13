using FiniteElementContainers
using Test
function test_structured_mesh()
  # testing bad input
  @test_throws ArgumentError StructuredMesh("bad element", (0., 0.), (1., 1.), (3, 3))
  @test_throws BoundsError StructuredMesh("tri3", (0., 0.), (0., 1.), (3, 3))

  # Quad4 test
  mesh = StructuredMesh("quad", (0., 0.), (1., 1.), (3, 3))
  coords = mesh.nodal_coords
  @test coords ≈ H1Field([
    0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0;
    0.0 0.0 0.0 0.5 0.5 0.5 1.0 1.0 1.0
  ])
  @test mesh.element_conns[:block_1] ≈ L2ElementField([
    1 4 2 5;
    2 5 3 6;
    5 8 6 9;
    4 7 5 8
  ])
  @test all(coords[2, mesh.nodeset_nodes[:bottom]] .≈ 0.)
  @test all(coords[1, mesh.nodeset_nodes[:right]] .≈ 1.)
  @test all(coords[2, mesh.nodeset_nodes[:top]] .≈ 1.)
  @test all(coords[1, mesh.nodeset_nodes[:left]] .≈ 0.)

  @test all(coords[2, mesh.sideset_nodes[:bottom]] .≈ 0.)
  @test all(coords[1, mesh.sideset_nodes[:right]] .≈ 1.)
  @test all(coords[2, mesh.sideset_nodes[:top]] .≈ 1.)
  @test all(coords[1, mesh.sideset_nodes[:left]] .≈ 0.)
  # tri3 test
  mesh = StructuredMesh("tri", (0., 0.), (1., 1.), (3, 3))
  @test mesh.nodal_coords ≈ H1Field([
    0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0;
    0.0 0.0 0.0 0.5 0.5 0.5 1.0 1.0  1.0
  ])
  @test mesh.element_conns[:block_1] ≈ L2ElementField([
    1 1 4 4 2 2 5 5;
    2 5 5 8 3 6 6 9;
    5 4 8 7 6 5 9 8
  ])
  @test all(coords[2, mesh.nodeset_nodes[:bottom]] .≈ 0.)
  @test all(coords[1, mesh.nodeset_nodes[:right]] .≈ 1.)
  @test all(coords[2, mesh.nodeset_nodes[:top]] .≈ 1.)
  @test all(coords[1, mesh.nodeset_nodes[:left]] .≈ 0.)

  @test all(coords[2, mesh.sideset_nodes[:bottom]] .≈ 0.)
  @test all(coords[1, mesh.sideset_nodes[:right]] .≈ 1.)
  @test all(coords[2, mesh.sideset_nodes[:top]] .≈ 1.)
  @test all(coords[1, mesh.sideset_nodes[:left]] .≈ 0.)

  @show mesh
end

# @testset ExtendedTestSet "Mesh" begin
#   test_structured_mesh()
# end

struct DummyMesh <: FiniteElementContainers.AbstractMesh
end

# @test ExtendedTestSet "Mesh definition" begin
function test_bad_mesh_methods()
  mesh = DummyMesh()
  @test_throws MethodError coordinates(mesh)
  @test_throws MethodError element_block_id_map(mesh, 1)
  @test_throws MethodError element_block_ids(mesh)
  @test_throws MethodError element_block_names(mesh)
  @test_throws MethodError element_connectivity(mesh, 1)
  @test_throws MethodError element_type(mesh, 1)
  @test_throws MethodError nodeset(mesh, 1)
  @test_throws MethodError nodesets(mesh, [1])
  @test_throws MethodError nodeset_ids(mesh)
  @test_throws MethodError nodeset_names(mesh)
  @test_throws MethodError num_dimensions(mesh)
  @test_throws MethodError num_nodes(mesh)
  @test_throws MethodError sideset(mesh, 1)
  @test_throws MethodError sidesets(mesh)
  @test_throws MethodError sideset_ids(mesh)
  @test_throws MethodError sideset_names(mesh)
end

function test_bad_mesh_file_type()
  file_name = "some_file.badext"
  @test_throws ErrorException UnstructuredMesh(file_name)
end

# test_bad_mesh_file_type()

@testset "Mesh" begin
  test_bad_mesh_file_type()
  test_bad_mesh_methods()
  test_structured_mesh()
end
