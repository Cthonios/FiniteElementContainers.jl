function test_non_allocated_function_space_volume(
  coords, conns, dof, ref_fe, 
  expected_element_vol, expected_vol
)
  q_degrees = [1, 2]
  for q_degree in q_degrees
    fspace = NonAllocatedFunctionSpace(dof, conns, q_degree, ref_fe)
    @show fspace
    for e in axes(conns, 2)
      @test expected_element_vol ≈ FiniteElementContainers.volume(fspace, coords, e)
    end
    @test expected_vol ≈ FiniteElementContainers.volume(fspace, coords)
  end

  # can't construct at the moment
  # q_degrees = [1, 2]
  # for q_degree in q_degrees
  #   fspace = VectorizedPreAllocatedFunctionSpace(dof, conns, q_degree, ref_fe, coords)
  #   @show fspace
  #   for e in axes(conns, 2)
  #     @test expected_element_vol ≈ FiniteElementContainers.volume(fspace, coords, e)
  #   end
  #   @test expected_vol ≈ FiniteElementContainers.volume(fspace, coords)
  # end
end

@testset ExtendedTestSet "FunctionSpaces" begin
  coords, conns = FiniteElementContainers.create_structured_mesh_data(7, 7, [0., 1.], [0., 1.])
  target_disp_grad = [
    0.1 0.4;
    -0.2 -0.1
  ]
  coords = NodalField{size(coords), Vector}(coords)
  conns = Connectivity{size(conns), Vector}(conns)
  dof = DofManager{Vector}(size(coords, 1), size(coords, 2))
  ref_fe = ReferenceFiniteElements.Tri3
  expected_vol = 1.0
  expected_element_vol = 0.5 / (6 * 6)
  test_non_allocated_function_space_volume(
    coords, conns, dof, ref_fe, 
    expected_element_vol, expected_vol
  )
end