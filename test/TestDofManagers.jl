
function test_dof_constructors(mesh)
  coords  = coordinates(mesh)
  coords  = NodalField{size(coords), Vector}(coords)
  dof1    = DofManager{1, size(coords, 2), Vector{Float64}}()
  dof2    = DofManager{Vector}(1, 16641)
  dof3    = DofManager(mesh, 1)
  @test dof1.unknown_dofs ≈ dof2.unknown_dofs
  @test dof1.unknown_dofs ≈ dof3.unknown_dofs
  dof1    = DofManager{2, size(coords, 2), Vector{Float64}}()
  dof2    = DofManager{Vector}(2, 16641)
  dof3    = DofManager(mesh, 2)
  @test dof1.unknown_dofs ≈ dof2.unknown_dofs
  @test dof1.unknown_dofs ≈ dof3.unknown_dofs
end

function test_dof_methods(mesh)
  # dof1 = DofManager(mesh, 2)
  # TODO do we add testing on a Matrix type or just deprecate that?
  dof1 = DofManager(mesh, 2)
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

@testset ExtendedTestSet "DofManager" begin
  mesh = FileMesh(ExodusDatabase, "./poisson/poisson.g")

  test_dof_constructors(mesh)
  test_dof_methods(mesh)
end
# bcs = [
  
# ]