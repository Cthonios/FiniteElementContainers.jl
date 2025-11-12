dummy_func_1(x, t) = 5. * t
dummy_func_2(x, t) = 5. * SVector{2, Float64}(1., 0.)

function test_dirichlet_bc_input()
  bc = DirichletBC(:my_var, :my_sset, dummy_func_1)
  @test bc.var_name == :my_var
  @test bc.sset_name == :my_sset
  @test typeof(bc.func) == typeof(dummy_func_1)
end

function test_dirichlet_bc_container_init()
  mesh = UnstructuredMesh("poisson/poisson.g")
  fspace = FunctionSpace(mesh, H1Field, Lagrange)
  u = VectorFunction(fspace, :displ)
  dof = DofManager(u)
  bc_in = DirichletBC(:displ_x, :sset_1, dummy_func_1)
  bc = FiniteElementContainers.DirichletBCContainer(mesh, dof, bc_in)
  @show bc
end

function test_neumann_bc_input()
  bc = NeumannBC(:my_var, :my_sset, dummy_func_2)
  @test bc.var_name == :my_var
  @test bc.sset_name == :my_sset
  @test typeof(bc.func) == typeof(dummy_func_2)
end

function test_neumann_bc_container_init()
  mesh = UnstructuredMesh("poisson/poisson.g")
  fspace = FunctionSpace(mesh, H1Field, Lagrange)
  u = VectorFunction(fspace, :displ)
  dof = DofManager(u)
  bc_in = NeumannBC(:displ, :sset_1, dummy_func_2)
  bc = FiniteElementContainers.NeumannBCContainer(mesh, dof, bc_in)
  @show bc
end

@testset ExtendedTestSet "BoundaryConditions" begin
  test_dirichlet_bc_input()
  test_dirichlet_bc_container_init()
  test_neumann_bc_input()
  # test_neumann_bc_container_init()
end
