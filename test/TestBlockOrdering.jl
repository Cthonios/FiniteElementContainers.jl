# Regression tests for element block ORDER.
#
# Meshes keep per-block data (`element_conns`, `element_id_maps`, ...) in `Dict`s
# keyed by block name, but the block names themselves live in an ordered
# `Vector`.  `Dict` iteration is hash order, so `values(mesh.element_conns)` and
# `mesh.element_block_names` are two *different* orderings of the same blocks.
# Anything that indexes blocks positionally has to pick one, and the two were
# mixed:
#
#   * `FunctionSpace` took connectivity from `values(mesh.element_conns)` but
#     block names and reference elements from `mesh.element_block_names`, so
#     `block_names(fspace)[b]` did not name the block whose elements sit at
#     index `b`.
#   * `BCBookKeeping` numbered blocks by `enumerate(values(mesh.element_id_maps))`
#     and then used that number to index `mesh.element_block_names`, so a side
#     set resolved to the wrong block name.
#   * `Parameters` paired `physics`/`properties` entry `b` with block `b` without
#     checking or reordering, so a `NamedTuple` written in any other order handed
#     each block another block's material.
#
# None of it was caught because every multi-block fixture in the suite has two
# blocks named `block_1`/`block_2`, for which the hash order happens to equal
# the file order.  Three blocks named `b1`/`b2`/`b3` do not have that property,
# which is what this fixture is for.
#
# The blocks are given DIFFERENT element counts (1, 2, 3) so that a permutation
# is visible as a size, not just as a name.

@testsnippet BlockOrderingHelper begin
  using Exodus
  using StaticArrays
  include("poisson/TestPoissonCommon.jl")

  # A 6 x 1 strip of QUAD4s over [0,6] x [0,1], split into three blocks holding
  # 1, 2 and 3 elements respectively.
  const BLOCK_NAMES = ["b1", "b2", "b3"]
  const BLOCK_SIZES = [1, 2, 3]

  function write_three_block_mesh(path)
    nx, ny = 6, 1
    coords = zeros(Float64, 2, (nx + 1) * (ny + 1))
    nid(i, j) = i + (j - 1) * (nx + 1)
    for j in 1:(ny + 1), i in 1:(nx + 1)
      coords[1, nid(i, j)] = float(i - 1)
      coords[2, nid(i, j)] = float(j - 1)
    end
    quad(i) = Int32[nid(i, 1), nid(i + 1, 1), nid(i + 1, 2), nid(i, 2)]

    isfile(path) && rm(path)
    init = Initialization(
      Int32(2), Int32(size(coords, 2)), Int32(nx),
      Int32(length(BLOCK_NAMES)), Int32(0), Int32(0)
    )
    exo = ExodusDatabase{Int32, Int32, Int32, Float64}(path, "w", init)
    write_coordinates(exo, coords)

    first_elem = 1
    for (b, (name, n)) in enumerate(zip(BLOCK_NAMES, BLOCK_SIZES))
      conn = reduce(hcat, [quad(i) for i in first_elem:(first_elem + n - 1)])
      write_block(exo, Block(Int32(b), size(conn, 2), 4, "QUAD4", conn))
      write_name(exo, Block(exo, b), name)
      first_elem += n
    end
    close(exo)
    return path
  end
end

@testitem "Block ordering - mesh and function space" setup=[BlockOrderingHelper] begin
  import FiniteElementContainers: block_names, block_conns, block_id_maps

  mktempdir() do dir
    mesh = UnstructuredMesh(write_three_block_mesh(joinpath(dir, "three_block.g")))

    # The premise of this fixture.  If a future Julia gives `Dict` insertion
    # order, the test below still passes but no longer exercises the bug, and
    # the block names should be re-chosen.
    if collect(keys(mesh.element_conns)) == mesh.element_block_names
      @warn "Dict order now matches file order; this fixture no longer " *
            "exercises the block ordering bug -- pick different block names."
    end

    @test block_names(mesh) == BLOCK_NAMES
    @test [size(c, 2) for c in block_conns(mesh)] == BLOCK_SIZES
    @test [length(m) for m in block_id_maps(mesh)] == BLOCK_SIZES

    V = FunctionSpace(mesh, H1Field, Lagrange)

    # The invariant: name, connectivity and reference element at index `b` all
    # belong to the same block.  Before the fix `block_names(V)` read
    # ["b1","b2","b3"] while `num_elements(V, b)` read [3, 2, 1].
    @test block_names(V) == BLOCK_NAMES
    @test [num_elements(V, b) for b in 1:length(BLOCK_NAMES)] == BLOCK_SIZES
    @test collect(keys(V.ref_fes)) == Symbol.(BLOCK_NAMES)
    for (b, name) in enumerate(BLOCK_NAMES)
      @test num_elements(V, b) == size(mesh.element_conns[name], 2)
    end
  end
end

@testitem "Block ordering - physics and properties alignment" setup=[BlockOrderingHelper] begin
  import FiniteElementContainers: block_names, _align_blocks, BlockMismatchError

  mktempdir() do dir
    mesh = UnstructuredMesh(write_three_block_mesh(joinpath(dir, "three_block.g")))
    V = FunctionSpace(mesh, H1Field, Lagrange)

    # A NamedTuple in any order is permuted into block order, and keeps its
    # association with the block it was named for.
    for order in ((b1 = 1, b2 = 2, b3 = 3),
                  (b3 = 3, b1 = 1, b2 = 2),
                  (b2 = 2, b3 = 3, b1 = 1))
      aligned = _align_blocks(V, order, "physics")
      @test keys(aligned) == (:b1, :b2, :b3)
      @test values(aligned) == (1, 2, 3)
    end

    # A single object is shared by every block, and is keyed by the real block
    # names rather than invented `region_N` placeholders.
    shared = _align_blocks(V, :one_material, "physics")
    @test keys(shared) == (:b1, :b2, :b3)
    @test all(v -> v === :one_material, values(shared))

    # A block with no entry, an entry naming no block, and an unnamed Tuple all
    # have to be rejected: each of them would otherwise silently give some block
    # another block's material.
    @test_throws BlockMismatchError _align_blocks(V, (b1 = 1, b2 = 2), "physics")
    @test_throws BlockMismatchError _align_blocks(V, (b1 = 1, b2 = 2, b3 = 3, b4 = 4), "physics")
    @test_throws BlockMismatchError _align_blocks(V, (b1 = 1, b2 = 2, typo = 3), "physics")
    @test_throws BlockMismatchError _align_blocks(V, (1, 2, 3), "physics")

    # The error has to name the blocks involved, otherwise it is no better than
    # the silent misassignment it replaces.
    err = try
      _align_blocks(V, (b1 = 1, b2 = 2, typo = 3), "physics")
    catch e
      sprint(showerror, e)
    end
    @test occursin("typo", err)
    @test occursin("b3", err)
  end
end

@testitem "Block ordering - parameters" setup=[BlockOrderingHelper] begin
  import FiniteElementContainers: block_names, block_size, BlockMismatchError

  mktempdir() do dir
    mesh = UnstructuredMesh(write_three_block_mesh(joinpath(dir, "three_block.g")))
    V = FunctionSpace(mesh, H1Field, Lagrange)
    u = ScalarFunction(V, "u")
    asm = SparseMatrixAssembler(u)

    f(X, _) = 0.0
    one_physics = Poisson(f)
    one_props = create_properties(one_physics)

    # Single material for the whole mesh: replicated, and keyed by block name.
    p = create_parameters(mesh, asm, one_physics, one_props)
    @test collect(keys(p.physics)) == Symbol.(BLOCK_NAMES)
    @test collect(keys(p.properties)) == Symbol.(BLOCK_NAMES)

    # State variables are allocated per block by walking `values(physics)`
    # against `block_quadrature_size(fspace, b)`, so their element counts are a
    # direct check that entry `b` really is block `b`.
    @test [block_size(p.state_old, b)[3] for b in 1:length(BLOCK_NAMES)] == BLOCK_SIZES

    # Per-block materials supplied out of order still land on the right blocks.
    scrambled_physics = (b3 = Poisson(f), b1 = Poisson(f), b2 = Poisson(f))
    scrambled_props = (b3 = one_props, b1 = one_props, b2 = one_props)
    p = create_parameters(mesh, asm, scrambled_physics, scrambled_props)
    @test collect(keys(p.physics)) == Symbol.(BLOCK_NAMES)
    @test collect(keys(p.properties)) == Symbol.(BLOCK_NAMES)
    @test [block_size(p.state_old, b)[3] for b in 1:length(BLOCK_NAMES)] == BLOCK_SIZES

    # And a mismatch stops the run instead of producing a plausible wrong answer.
    @test_throws BlockMismatchError create_parameters(
      mesh, asm, (b1 = Poisson(f), b2 = Poisson(f)), one_props
    )
    @test_throws BlockMismatchError create_parameters(
      mesh, asm, one_physics, (b1 = one_props, b2 = one_props, nope = one_props)
    )
  end
end
