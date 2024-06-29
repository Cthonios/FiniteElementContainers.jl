
function test_dof_constructors(mesh)
  coords  = coordinates(mesh)
  coords  = NodalField{size(coords), Vector}(coords)
  dof1    = DofManager{1, size(coords, 2), Vector{Float64}}()
  dof2    = DofManager{Vector{Float64}}(1, 16641)
  dof3    = DofManager(mesh, 1)
  @test dof1.unknown_dofs ≈ dof2.unknown_dofs
  @test dof1.unknown_dofs ≈ dof3.unknown_dofs
  dof1    = DofManager{2, size(coords, 2), Vector{Float64}}()
  dof2    = DofManager{Vector{Float64}}(2, 16641)
  dof3    = DofManager(mesh, 2)
  @test dof1.unknown_dofs ≈ dof2.unknown_dofs
  @test dof1.unknown_dofs ≈ dof3.unknown_dofs
end

function test_dof_methods()
  # dof1 = DofManager(mesh, 2)
  # TODO do we add testing on a Matrix type or just deprecate that?
  dof1 = DofManager{Matrix{Float64}}(2, 16641)
  @show typeof(dof1)
  @show dof1
  @test eltype(dof1) == Int64
  @test size(dof1) == (2, 16641)
  @test size(dof1, 1) == 2
  @test size(dof1, 2) == 16641
  @test FiniteElementContainers.num_dofs_per_node(dof1) == 2
  @test FiniteElementContainers.num_nodes(dof1) == 16641
  U = create_fields(dof1)
  @test size(U) == size(dof1)
  dof1 = DofManager{Vector{Float64}}(2, 16641)
  @show typeof(dof1)
  @show dof1
  @test eltype(dof1) == Int64
  @test size(dof1) == (2, 16641)
  @test size(dof1, 1) == 2
  @test size(dof1, 2) == 16641
  @test FiniteElementContainers.num_dofs_per_node(dof1) == 2
  @test FiniteElementContainers.num_nodes(dof1) == 16641
  U = create_fields(dof1)
  @test size(U) == size(dof1)
end

function test_bc_ids(mesh)
  nsets = nodeset.((mesh,), [1, 2, 3, 4])
  nsets = map(nset -> convert.(Int64, nset), nsets)
  dof = DofManager(mesh, 1)
  bc_dofs = FiniteElementContainers.bc_ids(dof, nsets, [1, 1, 1, 1])
end

function test_update_unknown_dofs(mesh)
  nsets = nodeset.((mesh,), [1, 2, 3, 4])
  nsets = map(nset -> convert.(Int64, nset), nsets)
  dof = DofManager(mesh, 1)
  bc_dofs = FiniteElementContainers.bc_ids(dof, nsets, [1, 1, 1, 1])
  update_unknown_dofs!(dof)
  @test size(dof.unknown_dofs) == (16641,)
  # here bc_nodes are the same as bc dofs
  update_unknown_dofs!(dof, bc_dofs)
  @test size(dof.unknown_dofs) == (16641 - length(bc_dofs),)
  dof = DofManager(mesh, 2)  
  bc_dofs = FiniteElementContainers.bc_ids(dof, nsets, [1, 1, 2, 2])
  update_unknown_dofs!(dof)
  @test size(dof.unknown_dofs) == (2 * 16641,)
  update_unknown_dofs!(dof, bc_dofs)
  @test size(dof.unknown_dofs) == (2 * 16641 - length(bc_dofs),)
end

function test_dof_correctness()
  dof = DofManager{Vector}(2, 10)
  update_unknown_dofs!(dof)
  update_unknown_dofs!(dof, [2, 4, 8, 11, 13])
  Uu = create_unknowns(dof)
  @test size(Uu) = (15,)
  U = create_fields(dof)
  @test size(U) == (2, 10)
  Uu .= rand(Float64, size(Uu))
  update_fields!(U, dof, Uu)
  @test U[dof.unknown_dofs] ≈ Uu
  @test FiniteElementContainers.num_unknowns(dof) == 15
  @test FiniteElementContainers.num_bcs(dof) == 5
end

@testset ExtendedTestSet "DofManager" begin
  mesh = FileMesh(ExodusDatabase, "./poisson/poisson.g")

  test_dof_constructors(mesh)
  test_dof_methods()
  test_bc_ids(mesh)
  test_update_unknown_dofs(mesh)
end
# bcs = [
  
# ]