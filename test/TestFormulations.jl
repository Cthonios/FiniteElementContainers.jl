function test_incompressible_plane_stress(∇N_X, ∇u_q, A_q)
  form = IncompressiblePlaneStress()
  @test FiniteElementContainers.num_dimensions(form) == 2
  ∇u_sm = modify_field_gradients(form, ∇u_q, SMatrix)
  @test ∇u_sm ≈ SMatrix{3, 3, Float64, 9}((
    1., 2., 0., 3., 4., 0., 0., 0., 1. / (-2.)
  ))
  ∇u_t = modify_field_gradients(form, ∇u_q, Tensor)
  @test ∇u_t ≈ Tensor{2, 3, Float64, 9}((
    1., 2., 0., 3., 4., 0., 0., 0., 1. / (-2.)
  ))
  ∇u_t = modify_field_gradients(form, Tensor{2, 2, Float64, 4}(∇u_q), Tensor)
  @test ∇u_t ≈ Tensor{2, 3, Float64, 9}((
    1., 2., 0., 3., 4., 0., 0., 0., 1. / (-2.)
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

  G = discrete_symmetric_gradient(form, ∇N_X)
  @test G[1, 1] ≈ ∇N_X[1, 1]
  @test G[1, 2] ≈ 0.0
  @test G[1, 3] ≈ ∇N_X[1, 2]
  # 
  @test G[2, 1] ≈ 0.0
  @test G[2, 2] ≈ ∇N_X[1, 2]
  @test G[2, 3] ≈ ∇N_X[1, 1]
  # 
  @test G[3, 1] ≈ ∇N_X[2, 1]
  @test G[3, 2] ≈ 0.0
  @test G[3, 3] ≈ ∇N_X[2, 2]
  # 
  @test G[4, 1] ≈ 0.0
  @test G[4, 2] ≈ ∇N_X[2, 2]
  @test G[4, 3] ≈ ∇N_X[2, 1]
  #
  @test G[5, 1] ≈ ∇N_X[3, 1]
  @test G[5, 2] ≈ 0.0
  @test G[5, 3] ≈ ∇N_X[3, 2]
  # 
  @test G[6, 1] ≈ 0.0
  @test G[6, 2] ≈ ∇N_X[3, 2]
  @test G[6, 3] ≈ ∇N_X[3, 1]
  #
  @test G[7, 1] ≈ ∇N_X[4, 1]
  @test G[7, 2] ≈ 0.0
  @test G[7, 3] ≈ ∇N_X[4, 2]
  # 
  @test G[8, 1] ≈ 0.0
  @test G[8, 2] ≈ ∇N_X[4, 2]
  @test G[8, 3] ≈ ∇N_X[4, 1]
end 

function test_plane_strain(∇N_X, ∇u_q, A_q)
  form = PlaneStrain()
  @test FiniteElementContainers.num_dimensions(form) == 2
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

  G = discrete_symmetric_gradient(form, ∇N_X)
  @test G[1, 1] ≈ ∇N_X[1, 1]
  @test G[1, 2] ≈ 0.0
  @test G[1, 3] ≈ ∇N_X[1, 2]
  # 
  @test G[2, 1] ≈ 0.0
  @test G[2, 2] ≈ ∇N_X[1, 2]
  @test G[2, 3] ≈ ∇N_X[1, 1]
  # 
  @test G[3, 1] ≈ ∇N_X[2, 1]
  @test G[3, 2] ≈ 0.0
  @test G[3, 3] ≈ ∇N_X[2, 2]
  # 
  @test G[4, 1] ≈ 0.0
  @test G[4, 2] ≈ ∇N_X[2, 2]
  @test G[4, 3] ≈ ∇N_X[2, 1]
  #
  @test G[5, 1] ≈ ∇N_X[3, 1]
  @test G[5, 2] ≈ 0.0
  @test G[5, 3] ≈ ∇N_X[3, 2]
  # 
  @test G[6, 1] ≈ 0.0
  @test G[6, 2] ≈ ∇N_X[3, 2]
  @test G[6, 3] ≈ ∇N_X[3, 1]
  #
  @test G[7, 1] ≈ ∇N_X[4, 1]
  @test G[7, 2] ≈ 0.0
  @test G[7, 3] ≈ ∇N_X[4, 2]
  # 
  @test G[8, 1] ≈ 0.0
  @test G[8, 2] ≈ ∇N_X[4, 2]
  @test G[8, 3] ≈ ∇N_X[4, 1]
end 

function test_three_dimensional(∇N_X, ∇u_q, A_q)
  form = ThreeDimensional()
  @test FiniteElementContainers.num_dimensions(form) == 3
  ∇u_sm = modify_field_gradients(form, ∇u_q, SMatrix)
  @test ∇u_q ≈ ∇u_sm
  ∇u_t = modify_field_gradients(form, ∇u_q, Tensor)
  @test ∇u_q ≈ ∇u_t
  ∇u_t = modify_field_gradients(form, Tensor{2, 3, Float64, 9}(∇u_q), Tensor)
  @test Tensor{2, 3, Float64, 9}(∇u_q) ≈ ∇u_t
  P_vec = extract_stress(form, ∇u_t)
  @test P_vec ≈ SVector{9, Float64}((1, 2, 3, 4, 5, 6, 7, 8, 9))
  #
  A_mat = extract_stiffness(form, A_q)
  @test A_mat[1, 1] ≈ 1.0
  @test A_mat[1, 2] ≈ 2.0
  @test A_mat[1, 3] ≈ 3.0
  @test A_mat[1, 4] ≈ 4.0
  @test A_mat[1, 5] ≈ 5.0
  @test A_mat[1, 6] ≈ 6.0
  @test A_mat[1, 7] ≈ 7.0
  @test A_mat[1, 8] ≈ 8.0
  @test A_mat[1, 9] ≈ 9.0
  #
  @test A_mat[2, 1] ≈ 10.0
  @test A_mat[2, 2] ≈ 11.0
  @test A_mat[2, 3] ≈ 12.0
  @test A_mat[2, 4] ≈ 13.0
  @test A_mat[2, 5] ≈ 14.0
  @test A_mat[2, 6] ≈ 15.0
  @test A_mat[2, 7] ≈ 16.0
  @test A_mat[2, 8] ≈ 17.0
  @test A_mat[2, 9] ≈ 18.0
  #
  @test A_mat[3, 1] ≈ 19.0
  @test A_mat[3, 2] ≈ 20.0
  @test A_mat[3, 3] ≈ 21.0
  @test A_mat[3, 4] ≈ 22.0
  @test A_mat[3, 5] ≈ 23.0
  @test A_mat[3, 6] ≈ 24.0
  @test A_mat[3, 7] ≈ 25.0
  @test A_mat[3, 8] ≈ 26.0
  @test A_mat[3, 9] ≈ 27.0
  #
  @test A_mat[4, 1] ≈ 28.0
  @test A_mat[4, 2] ≈ 29.0
  @test A_mat[4, 3] ≈ 30.0
  @test A_mat[4, 4] ≈ 31.0
  @test A_mat[4, 5] ≈ 32.0
  @test A_mat[4, 6] ≈ 33.0
  @test A_mat[4, 7] ≈ 34.0
  @test A_mat[4, 8] ≈ 35.0
  @test A_mat[4, 9] ≈ 36.0
  #
  @test A_mat[5, 1] ≈ 37.0
  @test A_mat[5, 2] ≈ 38.0
  @test A_mat[5, 3] ≈ 39.0
  @test A_mat[5, 4] ≈ 40.0
  @test A_mat[5, 5] ≈ 41.0
  @test A_mat[5, 6] ≈ 42.0
  @test A_mat[5, 7] ≈ 43.0
  @test A_mat[5, 8] ≈ 44.0
  @test A_mat[5, 9] ≈ 45.0
  #
  @test A_mat[6, 1] ≈ 46.0
  @test A_mat[6, 2] ≈ 47.0
  @test A_mat[6, 3] ≈ 48.0
  @test A_mat[6, 4] ≈ 49.0
  @test A_mat[6, 5] ≈ 50.0
  @test A_mat[6, 6] ≈ 51.0
  @test A_mat[6, 7] ≈ 52.0
  @test A_mat[6, 8] ≈ 53.0
  @test A_mat[6, 9] ≈ 54.0
  #
  @test A_mat[7, 1] ≈ 55.0
  @test A_mat[7, 2] ≈ 56.0
  @test A_mat[7, 3] ≈ 57.0
  @test A_mat[7, 4] ≈ 58.0
  @test A_mat[7, 5] ≈ 59.0
  @test A_mat[7, 6] ≈ 60.0
  @test A_mat[7, 7] ≈ 61.0
  @test A_mat[7, 8] ≈ 62.0
  @test A_mat[7, 9] ≈ 63.0
  #
  @test A_mat[8, 1] ≈ 64.0
  @test A_mat[8, 2] ≈ 65.0
  @test A_mat[8, 3] ≈ 66.0
  @test A_mat[8, 4] ≈ 67.0
  @test A_mat[8, 5] ≈ 68.0
  @test A_mat[8, 6] ≈ 69.0
  @test A_mat[8, 7] ≈ 70.0
  @test A_mat[8, 8] ≈ 71.0
  @test A_mat[8, 9] ≈ 72.0

  @test A_mat[9, 1] ≈ 73.0
  @test A_mat[9, 2] ≈ 74.0
  @test A_mat[9, 3] ≈ 75.0
  @test A_mat[9, 4] ≈ 76.0
  @test A_mat[9, 5] ≈ 77.0
  @test A_mat[9, 6] ≈ 78.0
  @test A_mat[9, 7] ≈ 79.0
  @test A_mat[9, 8] ≈ 80.0
  @test A_mat[9, 9] ≈ 81.0

  G = discrete_gradient(form, ∇N_X)
  for n in 1:3:24
    k = (n - 1) ÷ 3
    @test G[n, 1] ≈ ∇N_X[k + 1, 1]
    @test G[n, 2] ≈ 0.0
    @test G[n, 3] ≈ 0.0
    @test G[n, 4] ≈ ∇N_X[k + 1, 2]
    @test G[n, 5] ≈ 0.0
    @test G[n, 6] ≈ 0.0
    @test G[n, 7] ≈ ∇N_X[k + 1, 3]
    @test G[n, 8] ≈ 0.0
    @test G[n, 9] ≈ 0.0
    #
    @test G[n + 1, 1] ≈ 0.0
    @test G[n + 1, 2] ≈ ∇N_X[k + 1, 1]
    @test G[n + 1, 3] ≈ 0.0
    @test G[n + 1, 4] ≈ 0.0
    @test G[n + 1, 5] ≈ ∇N_X[k + 1, 2]
    @test G[n + 1, 6] ≈ 0.0
    @test G[n + 1, 7] ≈ 0.0
    @test G[n + 1, 8] ≈ ∇N_X[k + 1, 3]
    @test G[n + 1, 9] ≈ 0.0
    #
    @test G[n + 2, 1] ≈ 0.0
    @test G[n + 2, 2] ≈ 0.0
    @test G[n + 2, 3] ≈ ∇N_X[k + 1, 1]
    @test G[n + 2, 4] ≈ 0.0
    @test G[n + 2, 5] ≈ 0.0
    @test G[n + 2, 6] ≈ ∇N_X[k + 1, 2]
    @test G[n + 2, 7] ≈ 0.0
    @test G[n + 2, 8] ≈ 0.0
    @test G[n + 2, 9] ≈ ∇N_X[k + 1, 3]
  end

  G = discrete_symmetric_gradient(form, ∇N_X)
  for n in 1:3:24
    k = (n - 1) ÷ 3
    @test G[n, 1] ≈ ∇N_X[k + 1, 1]
    @test G[n, 2] ≈ 0.0
    @test G[n, 3] ≈ 0.0
    @test G[n, 4] ≈ ∇N_X[k + 1, 2]
    @test G[n, 5] ≈ 0.0
    @test G[n, 6] ≈ ∇N_X[k + 1, 3]
    #
    @test G[n + 1, 1] ≈ 0.0
    @test G[n + 1, 2] ≈ ∇N_X[k + 1, 2]
    @test G[n + 1, 3] ≈ 0.0
    @test G[n + 1, 4] ≈ ∇N_X[k + 1, 1]
    @test G[n + 1, 5] ≈ ∇N_X[k + 1, 3]
    @test G[n + 1, 6] ≈ 0.0
    #
    @test G[n + 2, 1] ≈ 0.0
    @test G[n + 2, 2] ≈ 0.0
    @test G[n + 2, 3] ≈ ∇N_X[k + 1, 3]
    @test G[n + 2, 4] ≈ 0.0
    @test G[n + 2, 5] ≈ ∇N_X[k + 1, 2]
    @test G[n + 2, 6] ≈ ∇N_X[k + 1, 1]
  end

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
  test_incompressible_plane_stress(∇N_X, ∇u_q, A_q)
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