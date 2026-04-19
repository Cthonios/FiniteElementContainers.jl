@testsnippet BCHelper begin
  dummy_func_1(x, t) = 5. * t
  dummy_func_2(x, t) = 5. * SVector{2, Float64}(1., 0.)
  dummy_func_3(x, t, u) = 5. * SVector{2, Float64}(1., 0.) + u

  mesh = UnstructuredMesh("poisson/poisson.g")
  fspace = FunctionSpace(mesh, H1Field, Lagrange)
end

@testitem "BCs - test_dirichlet_bc_input" setup=[BCHelper] begin
  bc = DirichletBC("my_var", dummy_func_1; block_name = "my_block")
  @test bc.block_name == "my_block"
  @test bc.nset_name === nothing
  @test bc.sset_name === nothing
  @test bc.var_name == "my_var"
  @test typeof(bc.func) == typeof(dummy_func_1)

  bc = DirichletBC("my_var", dummy_func_1; nodeset_name = "my_nset")
  @test bc.block_name === nothing
  @test bc.nset_name == "my_nset"
  @test bc.sset_name === nothing
  @test bc.var_name == "my_var"
  @test typeof(bc.func) == typeof(dummy_func_1)

  bc = DirichletBC("my_var", dummy_func_1; sideset_name = "my_sset")
  @test bc.block_name === nothing
  @test bc.nset_name === nothing
  @test bc.sset_name == "my_sset"
  @test bc.var_name == "my_var"
  @test typeof(bc.func) == typeof(dummy_func_1)
end

@testitem "BCs - test_dirichlet_bc_container_init" setup=[BCHelper] begin
  u = VectorFunction(fspace, "displ")
  dof = DofManager(u)
  bc_in = DirichletBC("displ_x", dummy_func_1; sideset_name = "sset_1")
  bcs = DirichletBCs(mesh, dof, DirichletBC[bc_in])
  @show bcs
end

@testitem "BCs - test_neumann_bc_input" setup=[BCHelper] begin
  bc = NeumannBC("my_var", dummy_func_1, "my_sset")
  @test bc.var_name == "my_var"
  @test bc.sset_name == "my_sset"
  @test typeof(bc.func) == typeof(dummy_func_1)
end

@testitem "BCs - test_neumann_bcs_init" setup=[BCHelper] begin
  u = VectorFunction(fspace, "displ")
  dof = DofManager(u)
  bc_in = NeumannBC("displ", dummy_func_2, "sset_1")
  bcs = NeumannBCs(mesh, dof, NeumannBC[bc_in])
  @show bcs
end

@testitem "BCs - test_periodic_bc_input" setup=[BCHelper] begin
  bc = PeriodicBC("my_var", "x", dummy_func_1, "my_sset_1", "my_sset_2")
  @test bc.var_name == "my_var"
  @test bc.direction == "x"
  @test bc.side_a_sset == "my_sset_1"
  @test bc.side_b_sset == "my_sset_2"
  @test typeof(bc.func) == typeof(dummy_func_1)
end

@testitem "BCs - test_robin_bc_input" setup=[BCHelper] begin
  bc = RobinBC("my_var", dummy_func_3, "my_sset")
  @test bc.var_name == "my_var"
  @test bc.sset_name == "my_sset"
  @test typeof(bc.func) == typeof(dummy_func_3)
end

@testitem "BCs - test_robin_bcs_init" setup=[BCHelper] begin
  u = VectorFunction(fspace, "displ")
  dof = DofManager(u)
  bc_in = RobinBC("displ", dummy_func_3, "sset_1")
  bcs = RobinBCs(mesh, dof, RobinBC[bc_in])
  @show bcs
end

@testitem "BCs - test_source_input" setup=[BCHelper] begin
  source = Source("my_var", dummy_func_1, "my_sset")
  @test source.var_name == "my_var"
  @test source.block_name == "my_sset"
  @test typeof(source.func) == typeof(dummy_func_1)
end

@testitem "BCs - test_sources_init" setup=[BCHelper] begin
  u = VectorFunction(fspace, "displ")
  dof = DofManager(u)
  source_in = Source("displ", dummy_func_2, "block_1")
  sources = Sources(mesh, dof, Source[source_in])
  @show sources
end
