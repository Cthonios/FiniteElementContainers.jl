function test_dof_constructors(fspace)
  # TODO this could be made more complicated
  u = ScalarFunction(fspace, :u)
  v = ScalarFunction(fspace, :v)
  w = VectorFunction(fspace, :w)
  dof = DofManager(u)
  dof = DofManager(u, v)
  dof = DofManager(u, v, w)
  @show dof
end

function test_dof_methods(fspace)
  u = VectorFunction(fspace, :u)
  dof1 = DofManager(u)
  @test eltype(dof1) == Int64
  @test FiniteElementContainers.num_dofs_per_node(dof1) == 2
  @test FiniteElementContainers.num_nodes(dof1) == 16641
  U = create_field(dof1, H1Field)
  @test size(U) == (2, 16641)
end

# function test_bc_ids(mesh)
#   nsets = nodeset.((mesh,), [1, 2, 3, 4])
#   nsets = map(nset -> convert.(Int64, nset), nsets)
#   dof = DofManager(mesh, 1)
#   bc_dofs = FiniteElementContainers.bc_ids(dof, nsets, [1, 1, 1, 1])
# end

# function test_update_unknown_dofs(mesh)
#   nsets = nodeset.((mesh,), [1, 2, 3, 4])
#   nsets = map(nset -> convert.(Int64, nset), nsets)
#   dof = DofManager(mesh, 1)
#   bc_dofs = FiniteElementContainers.bc_ids(dof, nsets, [1, 1, 1, 1])
#   update_unknown_dofs!(dof)
#   @test size(dof.unknown_dofs) == (16641,)
#   # here bc_nodes are the same as bc dofs
#   update_unknown_dofs!(dof, bc_dofs)
#   @test size(dof.unknown_dofs) == (16641 - length(bc_dofs),)
#   dof = DofManager(mesh, 2)  
#   bc_dofs = FiniteElementContainers.bc_ids(dof, nsets, [1, 1, 2, 2])
#   update_unknown_dofs!(dof)
#   @test size(dof.unknown_dofs) == (2 * 16641,)
#   update_unknown_dofs!(dof, bc_dofs)
#   @test size(dof.unknown_dofs) == (2 * 16641 - length(bc_dofs),)
# end

# function test_dof_correctness()
#   dof = DofManager{Vector{Float64}}(2, 10)
#   update_unknown_dofs!(dof)
#   update_unknown_dofs!(dof, [2, 4, 8, 11, 13])
#   Uu = create_unknowns(dof)
#   @test size(Uu) == (15,)
#   U = create_fields(dof)
#   @test size(U) == (2, 10)
#   Uu .= rand(Float64, size(Uu))
#   update_fields!(U, dof, Uu)
#   @test U[dof.unknown_dofs] ≈ Uu
#   @test FiniteElementContainers.num_unknowns(dof) == 15
#   @test FiniteElementContainers.num_bcs(dof) == 5
# end

@testset ExtendedTestSet "DofManager" begin
  # mesh = FileMesh(FiniteElementContainers.ExodusMesh, "./poisson/poisson.g")
  mesh = UnstructuredMesh("./poisson/poisson.g")
  fspace = FunctionSpace(mesh, H1Field, Lagrange)

  test_dof_constructors(fspace)
  test_dof_methods(fspace)
  # test_bc_ids(mesh)
  # test_update_unknown_dofs(mesh)
  # test_dof_correctness()
end
