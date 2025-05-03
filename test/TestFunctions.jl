@testset ExtendedTestSet "Functions" begin
  mesh = UnstructuredMesh("poisson/poisson.g")
  fspace = FunctionSpace(mesh, H1Field, Lagrange)

  u = ScalarFunction(fspace, :u)
  @show u
  @test length(u) == 1
  @test names(u) == (:u,)

  u = VectorFunction(fspace, :u)
  @show u
  @test length(u) == 2
  @test names(u) == (:u_x, :u_y)
  
  u = TensorFunction(fspace, :u)
  @show u
  @test length(u) == 9
  @test names(u) == (:u_xx, :u_yy, :u_zz, :u_yz, :u_xz, :u_xy, :u_zy, :u_zx, :u_yx)

  u = TensorFunction(fspace, :u; use_spatial_dimension=true)
  @show u
  @test length(u) == 4
  @test names(u) == (:u_xx, :u_yy, :u_xy, :u_yx)

  u = SymmetricTensorFunction(fspace, :u)
  @show u
  @test length(u) == 6
  @test names(u) == (:u_xx, :u_yy, :u_zz, :u_yz, :u_xz, :u_xy)

  u = SymmetricTensorFunction(fspace, :u; use_spatial_dimension=true)
  @show u
  @test length(u) == 3
  @test names(u) == (:u_xx, :u_yy, :u_xy)
end
