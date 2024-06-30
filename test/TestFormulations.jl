function test_plane_strain(∇N_X, ∇u_q, A_q)
  form = PlaneStrain()
  ∇u_sm = modify_field_gradients(form, ∇u_q, SMatrix)
  @test ∇u_sm ≈ SMatrix{3, 3, Float64, 9}((
    1., 2., 0., 3., 4., 0., 0., 0., 0.
  ))
  ∇u_t = modify_field_gradients(form, ∇u_q, Tensor)
  @test ∇u_t ≈ Tensor{2, 3, Float64, 9}((
    1., 2., 0., 3., 4., 0., 0., 0., 0.
  ))
  ∇u_t = modify_field_gradients(form, Tensor{2, 2, Float64, 4}(∇u_q), Tensor)
  @test ∇u_t ≈ Tensor{2, 3, Float64, 9}((
    1., 2., 0., 3., 4., 0., 0., 0., 0.
  ))
  P_vec = extract_stress(form, ∇u_t)
  @test P_vec ≈ SVector{4, Float64}((1., 2., 3., 4.))
  A_mat = extract_stiffness(form, A_q)
  @test A_mat[1, 1] ≈ 1.0
  @test A_mat[1, 2] ≈ 2.0
  @test A_mat[1, 3] ≈ 4.0
  @test A_mat[1, 4] ≈ 5.0
  #
  @test A_mat[2, 1] ≈ 10.0
  @test A_mat[2, 2] ≈ 11.0
  @test A_mat[2, 3] ≈ 13.0
  @test A_mat[2, 4] ≈ 14.0
  #
  @test A_mat[3, 1] ≈ 28.0
  @test A_mat[3, 2] ≈ 29.0
  @test A_mat[3, 3] ≈ 31.0
  @test A_mat[3, 4] ≈ 32.0
  #
  @test A_mat[4, 1] ≈ 37.0
  @test A_mat[4, 2] ≈ 38.0
  @test A_mat[4, 3] ≈ 40.0
  @test A_mat[4, 4] ≈ 41.0

  G = discrete_gradient(form, ∇N_X)
  @test G[1, 1] ≈ ∇N_X[1, 1]
  @test G[1, 2] ≈ 0.0
  @test G[1, 3] ≈ ∇N_X[1, 2]
  @test G[1, 4] ≈ 0.0
  #
  @test G[2, 1] ≈ 0.0
  @test G[2, 2] ≈ ∇N_X[1, 1]
  @test G[2, 3] ≈ 0.0
  @test G[2, 4] ≈ ∇N_X[1, 2]
  #
  @test G[3, 1] ≈ ∇N_X[2, 1]
  @test G[3, 2] ≈ 0.0
  @test G[3, 3] ≈ ∇N_X[2, 2]
  @test G[3, 4] ≈ 0.0
  #
  @test G[4, 1] ≈ 0.0
  @test G[4, 2] ≈ ∇N_X[2, 1]
  @test G[4, 3] ≈ 0.0
  @test G[4, 4] ≈ ∇N_X[2, 2]
  # half
  @test G[5, 1] ≈ ∇N_X[3, 1]
  @test G[5, 2] ≈ 0.0
  @test G[5, 3] ≈ ∇N_X[3, 2]
  @test G[5, 4] ≈ 0.0
  #
  @test G[6, 1] ≈ 0.0
  @test G[6, 2] ≈ ∇N_X[3, 1]
  @test G[6, 3] ≈ 0.0
  @test G[6, 4] ≈ ∇N_X[3, 2]
  #
  @test G[7, 1] ≈ ∇N_X[4, 1]
  @test G[7, 2] ≈ 0.0
  @test G[7, 3] ≈ ∇N_X[4, 2]
  @test G[7, 4] ≈ 0.0
  #
  @test G[8, 1] ≈ 0.0
  @test G[8, 2] ≈ ∇N_X[4, 1]
  @test G[8, 3] ≈ 0.0
  @test G[8, 4] ≈ ∇N_X[4, 2]
end 

function test_three_dimensional(∇N_X, ∇u_q, A_q)
  form = ThreeDimensional()
  ∇u_sm = modify_field_gradients(form, ∇u_q, SMatrix)
  @test ∇u_q ≈ ∇u_sm
  ∇u_t = modify_field_gradients(form, ∇u_q, Tensor)
  @test ∇u_q ≈ ∇u_t
  ∇u_t = modify_field_gradients(form, Tensor{2, 3, Float64, 9}(∇u_q), Tensor)
  @test Tensor{2, 3, Float64, 9}(∇u_q) ≈ ∇u_t
  P_vec = extract_stress(form, ∇u_t)
  @test P_vec ≈ SVector{9, Float64}((1, 2, 3, 4, 5, 6, 7, 8, 9))
end

# TODO test formulations on various element types
@testset ExtendedTestSet "Formulations" begin
  X = [
    0.0 0.0;
    1.0 0.0;
    1.0 1.0; 
    0.0 1.0
  ]'
  ref_fe = ReferenceFE(Quad4(1))
  ∇N_X = FiniteElementContainers.map_shape_function_gradients(X, shape_function_gradients(ref_fe, 1))
  ∇u_q = SMatrix{2, 2, Float64, 4}((1., 2., 3., 4.))
  A_q = Tensor{4, 3, Float64, 81}(reshape(1:81, 9, 9)')
  test_plane_strain(∇N_X, ∇u_q, A_q)
  X = [
    0.0 0.0 0.0;
    1.0 0.0 0.0;
    1.0 1.0 0.0; 
    0.0 1.0 0.0;
    0.0 0.0 1.0;
    1.0 0.0 1.0;
    1.0 1.0 1.0; 
    0.0 1.0 1.0;
  ]'
  ref_fe = ReferenceFE(Hex8(1))
  ∇N_X = FiniteElementContainers.map_shape_function_gradients(X, shape_function_gradients(ref_fe, 1))
  ∇u_q = SMatrix{3, 3, Float64, 9}((1., 2., 3., 4., 5., 6., 7., 8., 9.))
  test_three_dimensional(∇N_X, ∇u_q, A_q)
end