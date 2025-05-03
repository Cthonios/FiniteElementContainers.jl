dummy_func(x, t) = 5. * t

function test_dirichlet_bc_input()
  bc = DirichletBC(:my_var, :my_sset, dummy_func)
  @test bc.var_name == :my_var
  @test bc.sset_name == :my_sset
  @test typeof(bc.func) == typeof(dummy_func)
end

@testset ExtendedTestSet "BoundaryConditions" begin
  test_dirichlet_bc_input()
end
