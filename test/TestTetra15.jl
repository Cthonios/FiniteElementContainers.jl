# TETRA15: the quadratic tetrahedron enriched with a node at the centroid of
# each face and one at the element centroid (Tet{EnrichedLagrange, 2} of
# ReferenceFiniteElements).  Node order of the Exodus library: 1-4 vertices,
# 5-10 edge midpoints (edges 1-2, 2-3, 1-3, 1-4, 2-4, 3-4), 11 on face 1-3-2
# (side 4), 12 on face 2-3-4 (side 2), 13 on face 1-4-3 (side 3), 14 on face
# 1-2-4 (side 1), 15 centroid.
#
# ex_get_side_set_node_list of the Exodus library reports seven nodes per
# side for this element but fills only six; the reader has to supply the
# seventh from the connectivity, and these tests check that it does.

@testsnippet Tetra15Helper begin
  using Exodus

  # Two tetrahedra sharing the face 1-2-3 of the first (vertices 1, 2, 3 and
  # apexes 4 and 5).  Each element block holds one element, so side-set
  # element numbers cross a block boundary.
  # `sideset_elems` selects the elements whose four sides form the side set;
  # a side set may not mix elements with different numbers of side nodes.
  function write_two_tet15_mesh(path; types = ("TETRA15", "TETRA15"), sideset_elems = (1, 2))
    V = [0.0 1.0 0.0 0.0 0.0;
         0.0 0.0 1.0 0.0 0.0;
         0.0 0.0 0.0 1.0 -1.0]
    edges = ((1, 2), (2, 3), (1, 3), (1, 4), (2, 4), (3, 4))
    faces = ((1, 3, 2), (2, 3, 4), (1, 4, 3), (1, 2, 4))
    elems = ([1, 2, 3, 4], [1, 3, 2, 5])   # both positively oriented
    coords = [V[:, i] for i in 1:5]
    node_of = Dict{Vector{Int}, Int}()
    newnode!(key, x) = get!(node_of, key) do
      push!(coords, x); length(coords)
    end
    conns = Matrix{Int32}[]
    for (b, v) in enumerate(elems)
      c = zeros(Int32, 15)
      c[1:4] .= v
      for (k, (p, q)) in enumerate(edges)
        c[4 + k] = newnode!(sort([v[p], v[q]]), (coords[v[p]] .+ coords[v[q]]) ./ 2)
      end
      for (f, (p, q, r)) in enumerate(faces)
        c[10 + f] = newnode!(sort([v[p], v[q], v[r]]),
                             (coords[v[p]] .+ coords[v[q]] .+ coords[v[r]]) ./ 3)
      end
      c[15] = newnode!([-b], sum(coords[v[i]] for i in 1:4) ./ 4)
      nn = types[b] == "TETRA15" ? 15 : 10
      push!(conns, reshape(c[1:nn], nn, 1))
    end
    X = reduce(hcat, coords)
    isfile(path) && rm(path)
    init = Initialization{Int32}(3, size(X, 2), 2, 2, 0, 1)
    exo = ExodusDatabase{Int32, Int32, Int32, Float64}(path, "w", init)
    write_coordinates(exo, X)
    write_names(exo, Block, ["lower", "upper"])
    for b in 1:2
      write_block(exo, b, types[b], conns[b])
    end
    # every side of the selected elements
    ss_elems = Int32[e for e in sideset_elems for _ in 1:4]
    ss_sides = Int32[s for _ in sideset_elems for s in 1:4]
    write_set(exo, SideSet(Int32(1), ss_elems, ss_sides, Int32[], Int32[]))
    write_names(exo, SideSet, ["all"])
    close(exo)
    return path, conns
  end
end

@testitem "TETRA15 - side sets carry the face node of each side" setup=[Tetra15Helper] begin
  mktempdir() do dir
    path, conns = write_two_tet15_mesh(joinpath(dir, "two_tet15.g"))
    mesh = UnstructuredMesh(path)
    @test mesh.element_types["lower"] == "TETRA15"
    @test size(mesh.element_conns["upper"]) == (15, 1)

    n_nodes = size(mesh.nodal_coords, 2)
    nodes = mesh.sideset_nodes["all"]
    @test length(nodes) == 8 * 7
    @test all(1 .<= nodes .<= n_nodes)

    side_nodes = reshape(mesh.sideset_side_nodes["all"], 7, 8)
    face_node = (14, 12, 13, 11)
    for k in 1:8
      e = mesh.sideset_elems["all"][k]
      s = mesh.sideset_sides["all"][k]
      @test side_nodes[7, k] == conns[e][face_node[s], 1]
      # the six other nodes are the ones the library fills
      @test Set(side_nodes[1:6, k]) ⊆ Set(conns[e][1:10, 1])
    end
    # the shared face has one node, listed from both elements
    @test conns[1][11, 1] == conns[2][11, 1]
  end
end

@testitem "TETRA15 - the function space uses the enriched basis" setup=[Tetra15Helper] begin
  using ReferenceFiniteElements
  mktempdir() do dir
    path, _ = write_two_tet15_mesh(joinpath(dir, "two_tet15.g"))
    mesh = UnstructuredMesh(path)
    V = FunctionSpace(mesh, H1Field, Lagrange, GaussLegendre)
    for name in (:lower, :upper)
      re = V.ref_fes[name]
      @test re.element == Tet{EnrichedLagrange, 2}()
      @test num_cell_quadrature_points(re) == 14
      @test num_cell_dofs(re) == 15
    end
    # passing the enriched type explicitly is the same request
    V2 = FunctionSpace(mesh, H1Field, EnrichedLagrange, GaussLegendre)
    @test V2.ref_fes[:lower].element == Tet{EnrichedLagrange, 2}()
    # any other basis is refused
    @test_throws ErrorException FunctionSpace(mesh, H1Field, Hermite, GaussLegendre)
  end
end

@testitem "TETRA15 - defaults are per block" setup=[Tetra15Helper] begin
  using ReferenceFiniteElements
  mktempdir() do dir
    # a TETRA10 block before the TETRA15 block: each gets its own degree
    path, conns = write_two_tet15_mesh(joinpath(dir, "mixed.g");
                                       types = ("TETRA10", "TETRA15"), sideset_elems = (2,))
    mesh = UnstructuredMesh(path)
    # the side set lies in the second block, so the side-node fill crosses
    # the block offset
    side_nodes = reshape(mesh.sideset_side_nodes["all"], 7, 4)
    @test side_nodes[7, :] == conns[2][[14, 12, 13, 11], 1]
    V = FunctionSpace(mesh, H1Field, Lagrange, GaussLegendre)
    @test V.ref_fes[:lower].element == Tet{Lagrange, 2}()
    @test num_cell_quadrature_points(V.ref_fes[:lower]) == 4
    @test V.ref_fes[:upper].element == Tet{EnrichedLagrange, 2}()
    @test num_cell_quadrature_points(V.ref_fes[:upper]) == 14
  end
end
