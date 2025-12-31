using FiniteElementContainers
using Test

function test_amr_mesh()
  mesh = StructuredMesh("tri", (0., 0.), (1., 1.), (5, 5))
  amr = FiniteElementContainers.AMRMesh(mesh)

  @test length(unique(amr.edges)) == length(amr.edges)

  # test edge adj validity
  for (el, er) in amr.edge2adjelem[:block_1]
    @test el > 0
    @test er >= -1
  end
  is_boundary_edge(e) = edge2elem[e][2] == -1

  for e in axes(amr.elem2edge[:block_1], 2)
    for le in axes(amr.elem2edge[:block_1], 1)
      eid = amr.elem2edge[:block_1][le, e]
      @test e in amr.edge2adjelem[:block_1][eid]
    end
  end

  @test all(1 .≤ amr.ref_edges[:block_1] .≤ 3)

  for e in axes(values(amr.element_conns)[1], 2)
    gid = amr.elem2edge[:block_1][amr.ref_edges[:block_1][e], e]
    @test gid > 0
  end

  
  # some simple refinement tests
  coords = H1Field([
    0.0 1.0 0.0;
    0.0 0.0 1.0
  ])

  conns = L2ElementField{Int64, Vector{Int64}, 3}([1, 2, 3])
  ref_edge = [3]  # edge (1,2)
  # elem2edge, edge2elem = build_edges(conns)
  edge_dict = Dict{NTuple{2, Int}, Int}()
  edges, elem2edge, edge2adjelem = FiniteElementContainers._create_edges!(edge_dict, conns)

  edge_midpoint = Dict{Int, Int}()

  nodeset_nodes = Dict(:nset_all => [1, 2, 3])
  FiniteElementContainers._refine_element!(
      coords, conns, ref_edge,
      # elem2edge, edge_midpoint,
      elem2edge, edge2adjelem, 
      nodeset_nodes,
      edge_midpoint,
      1
  )

  @test size(conns) == (3, 2)
  @test size(coords, 2) == 4
  @test length(nodeset_nodes[:nset_all]) == 4
  # @test sum(triangle_area.(conns)) ≈ 0.5

  refine = 1:size(amr.element_conns[:block_1], 2) |> collect
  refine = [1, 5, 7]
  FiniteElementContainers._refine!(amr, refine)
  FiniteElementContainers.write_to_file(amr, "atri.exo"; force = true)
end

function test_structured_mesh()
  # testing bad input
  @test_throws ArgumentError StructuredMesh("bad element", (0., 0.), (1., 1.), (3, 3))
  @test_throws BoundsError StructuredMesh("tri3", (0., 0.), (0., 1.), (3, 3))

  # Hex8 test
  mesh = StructuredMesh("hex", (0., 0., 0.), (1., 1., 1.), (3, 3, 3))
  coords = mesh.nodal_coords
  @test all(coords[2, mesh.nodeset_nodes[:bottom]] .≈ 0.)
  @test all(coords[1, mesh.nodeset_nodes[:right]] .≈ 1.)
  @test all(coords[3, mesh.nodeset_nodes[:front]] .≈ 1.)
  @test all(coords[2, mesh.nodeset_nodes[:top]] .≈ 1.)
  @test all(coords[1, mesh.nodeset_nodes[:left]] .≈ 0.)
  @test all(coords[3, mesh.nodeset_nodes[:back]] .≈ 0.)

  # currently failing TODO fix this
  # @test all(coords[2, mesh.sideset_nodes[:bottom]] .≈ 0.)
  # @test all(coords[1, mesh.sideset_nodes[:right]] .≈ 1.)
  # @test all(coords[3, mesh.sideset_nodes[:front]] .≈ 1.)
  # @test all(coords[2, mesh.sideset_nodes[:top]] .≈ 1.)
  # @test all(coords[1, mesh.sideset_nodes[:left]] .≈ 0.)
  # @test all(coords[3, mesh.sideset_nodes[:right]] .≈ 0.)

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

function test_write_mesh()
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

@testset "Mesh" begin
  test_amr_mesh()
  test_bad_mesh_file_type()
  test_bad_mesh_methods()
  test_structured_mesh()
  test_write_mesh()
end
