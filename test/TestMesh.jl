@testitem "Meshes - test_gmsh_mesh" tags=[:gmsh] begin
  using Gmsh
  geo_file_tri3 = Base.source_dir() * "/gmsh/square_meshed_with_tris.geo"
  msh_file_tri3 = Base.source_dir() * "/gmsh/square_meshed_with_tris.msh"
  mesh = UnstructuredMesh(geo_file_tri3)
  mesh = UnstructuredMesh(msh_file_tri3)
end

@testitem "Meshes - test_structured_mesh" begin
  # testing bad input
  @test_throws ArgumentError StructuredMesh("bad element", (0., 0.), (1., 1.), (3, 3))
  @test_throws BoundsError StructuredMesh("tri3", (0., 0.), (0., 1.), (3, 3))

  # Hex8 test
  mesh = StructuredMesh("hex", (0., 0., 0.), (1., 1., 1.), (3, 3, 3))
  coords = mesh.nodal_coords
  @test all(coords[2, mesh.nodeset_nodes["bottom"]] .≈ 0.)
  @test all(coords[1, mesh.nodeset_nodes["right"]] .≈ 1.)
  @test all(coords[3, mesh.nodeset_nodes["front"]] .≈ 1.)
  @test all(coords[2, mesh.nodeset_nodes["top"]] .≈ 1.)
  @test all(coords[1, mesh.nodeset_nodes["left"]] .≈ 0.)
  @test all(coords[3, mesh.nodeset_nodes["back"]] .≈ 0.)

  # currently failing TODO fix this
  # @test all(coords[2, mesh.sideset_nodes["bottom"]] .≈ 0.)
  # @test all(coords[1, mesh.sideset_nodes["right"]] .≈ 1.)
  # @test all(coords[3, mesh.sideset_nodes[:front]] .≈ 1.)
  # @test all(coords[2, mesh.sideset_nodes["top"]] .≈ 1.)
  # @test all(coords[1, mesh.sideset_nodes["left"]] .≈ 0.)
  # @test all(coords[3, mesh.sideset_nodes["right"]] .≈ 0.)

  # Quad4 test
  mesh = StructuredMesh("quad", (0., 0.), (1., 1.), (3, 3))
  coords = mesh.nodal_coords
  @test coords ≈ H1Field([
    0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0;
    0.0 0.0 0.0 0.5 0.5 0.5 1.0 1.0 1.0
  ])
  @test mesh.element_conns["block_1"] ≈ [
    1 4 2 5;
    2 5 3 6;
    5 8 6 9;
    4 7 5 8
  ]
  @test all(coords[2, mesh.nodeset_nodes["bottom"]] .≈ 0.)
  @test all(coords[1, mesh.nodeset_nodes["right"]] .≈ 1.)
  @test all(coords[2, mesh.nodeset_nodes["top"]] .≈ 1.)
  @test all(coords[1, mesh.nodeset_nodes["left"]] .≈ 0.)

  @test all(coords[2, mesh.sideset_nodes["bottom"]] .≈ 0.)
  @test all(coords[1, mesh.sideset_nodes["right"]] .≈ 1.)
  @test all(coords[2, mesh.sideset_nodes["top"]] .≈ 1.)
  @test all(coords[1, mesh.sideset_nodes["left"]] .≈ 0.)
  # tri3 test
  mesh = StructuredMesh("tri", (0., 0.), (1., 1.), (3, 3))
  @test mesh.nodal_coords ≈ H1Field([
    0.0 0.5 1.0 0.0 0.5 1.0 0.0 0.5 1.0;
    0.0 0.0 0.0 0.5 0.5 0.5 1.0 1.0  1.0
  ])
  @test mesh.element_conns["block_1"] ≈ [
    1 1 4 4 2 2 5 5;
    2 5 5 8 3 6 6 9;
    5 4 8 7 6 5 9 8
  ]
  @test all(coords[2, mesh.nodeset_nodes["bottom"]] .≈ 0.)
  @test all(coords[1, mesh.nodeset_nodes["right"]] .≈ 1.)
  @test all(coords[2, mesh.nodeset_nodes["top"]] .≈ 1.)
  @test all(coords[1, mesh.nodeset_nodes["left"]] .≈ 0.)

  @test all(coords[2, mesh.sideset_nodes["bottom"]] .≈ 0.)
  @test all(coords[1, mesh.sideset_nodes["right"]] .≈ 1.)
  @test all(coords[2, mesh.sideset_nodes["top"]] .≈ 1.)
  @test all(coords[1, mesh.sideset_nodes["left"]] .≈ 0.)

  @show mesh
end

@testitem "Meshes - test_write_mesh" begin
  # show below is to cover Base.show in tests
  @show mesh = StructuredMesh("hex", (0., 0., 0.), (1., 1., 1.), (3, 3, 3))
  FiniteElementContainers.write_to_file(mesh, "shex.exo")
  rm("shex.exo", force = true)

  mesh = StructuredMesh("quad", (0., 0.), (1., 1.), (3, 3))
  FiniteElementContainers.write_to_file(mesh, "squad.exo")
  rm("squad.exo", force = true)

  mesh = StructuredMesh("tri", (0., 0.), (1., 1.), (3, 3))
  FiniteElementContainers.write_to_file(mesh, "stri.exo")
  rm("stri.exo", force = true)

  # TODO eventually put an exodiff test below
  dir = Base.source_dir()
  mesh = UnstructuredMesh("$dir/poisson/poisson.g")
  FiniteElementContainers.write_to_file(mesh, "umesh.exo")
  rm("umesh.exo")
end

@testitem "Meshes - test_bad_mesh_file_type" begin
  file_name = "some_file.badext"
  @test_throws ErrorException UnstructuredMesh(file_name)
end

@testitem "Meshes - topology" begin
  import FiniteElementContainers as FEC
  file_name = "poisson/poisson.g"
  mesh = UnstructuredMesh(file_name)
  topology = FEC.MeshTopology(mesh)
end

@testitem "Meshes - coloring" begin
  import FiniteElementContainers as FEC
  file_name = "poisson/poisson.g"
  mesh = UnstructuredMesh(file_name)
  coloring = FEC.MeshColoring(mesh)
  @test FEC.validate_coloring(mesh, coloring)

  coloring = FEC.MeshColoring(mesh; share_across_blocks = false)
  @test FEC.validate_coloring(mesh, coloring)
end
