function test_plane_strain(∇N_x, ∇u_q, A_q)
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
  # A_mat = extract_stiffness(form, A_q)
  # @test A_mat[1, 1] ≈ 1.0
  # @test A_mat[1, 2] ≈ 9.0
  # @test A_mat[1, 3] ≈ 6.0
  # @test A_mat[1, 4] ≈ 2.0
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