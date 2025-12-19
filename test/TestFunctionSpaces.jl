function test_non_allocated_function_space_volume(
  coords, elem_id_map, conns, dof, ref_fe, 
  expected_element_vol, expected_vol
)
  q_degrees = [1, 2]
  for q_degree in q_degrees
    fspace = NonAllocatedFunctionSpace(dof, elem_id_map, conns, q_degree, ref_fe)
    @show fspace
    for e in axes(conns, 2)
      @test expected_element_vol ≈ FiniteElementContainers.volume(fspace, coords, e)
    end
    @test expected_vol ≈ FiniteElementContainers.volume(fspace, coords)
  end

  # can't construct at the moment
  # q_degrees = [1, 2]
  # for q_degree in q_degrees
  #   fspace = VectorizedPreAllocatedFunctionSpace(dof, elem_id_map, conns, q_degree, "TRI3", coords)
  #   fspace = VectorizedPreAllocatedFunctionSpace(dof, elem_id_map, conns, q_degree, ref_fe, coords)
  #   @show fspace
  #   for e in axes(conns, 2)
  #     @test expected_element_vol ≈ FiniteElementContainers.volume(fspace, coords, e)
  #   end
  #   @test expected_vol ≈ FiniteElementContainers.volume(fspace, coords)
  # end
end

# function test_element_level_field_methods(coords, elem_id_map, conns, dof, ref_fe)

#   q_degrees = [1, 2]
#   U = create_fields(dof)
#   U .= rand(Float64, size(U))
#   for q_degree in q_degrees
#     fspace = VectorizedPreAllocatedFunctionSpace(dof, elem_id_map, conns, q_degree, ref_fe, coords)
#     U_els = element_level_fields(fspace, U)
#     for e in 1:num_elements(fspace)
#       @test U_els[e] ≈ element_level_fields(fspace, U, e)
#     end
#   end
# end

function test_linear_reproducing(coords, elem_id_map, conns, dof, ref_fe, target_disp_grad)
  q_degrees = [1]
  U = similar(coords)
  for n in axes(coords, 2)
    # U[:, n] = target_disp_grad' * coords[:, n]
    U[:, n] = (coords[:, n]' * target_disp_grad)'
  end

  for q_degree in q_degrees
    fspace = NonAllocatedFunctionSpace(dof, elem_id_map, conns, q_degree, ref_fe)
    for e in 1:num_elements(fspace)
      # X_el = element_level_coordinates(fspace, coords, e)
      # U_el = element_level_fields(fspace, U, e)
      for q in 1:FiniteElementContainers.num_q_points(fspace)
        U_q = quadrature_level_field_values(fspace, coords, U, q, e)
        display(U_q)
      end
    end
  end
end

# @testset "FunctionSpaces" begin
#   coords, conns = FiniteElementContainers.create_structured_mesh_data(7, 7, [0., 1.], [0., 1.])
#   target_disp_grad = [
#     0.1 0.4;
#     -0.2 -0.1
#   ]
#   coords = H1Field(coords)

#   # coords = NodalField{size(coords)}(coords)
#   # elem_id_map = Dict{Int, Int}(zip(1:size(conns, 2), 1:size(conns, 2)))
#   # conns = Connectivity{size(conns)}(conns)
#   # dof = DofManager{Vector{Float64}}(size(coords, 1), size(coords, 2))
#   # ref_fe = ReferenceFiniteElements.Tri3
#   # expected_vol = 1.0
#   # expected_element_vol = 0.5 / (6 * 6)
#   # # test_element_level_field_methods(
#   # #   coords, elem_id_map, conns, dof, ref_fe
#   # # )
#   # test_non_allocated_function_space_volume(
#   #   coords, elem_id_map, conns, dof, ref_fe, 
#   #   expected_element_vol, expected_vol
#   # )
#   # # test_linear_reproducing(
#   # #   coords, elem_id_map, conns, dof, ref_fe, 
#   # #   target_disp_grad
#   # # )
# end


function test_bad_interp_type(mesh)
  @test_throws MethodError FunctionSpace(mesh, H1Field, :SomethingNotSupported)
end

function test_fspace_h1_field(mesh)
  @show fspace = FunctionSpace(mesh, H1Field, Lagrange)
end

function test_fspace_l2_element_field(mesh)
  @show fspace = FunctionSpace(mesh, L2ElementField, Lagrange)
end

function test_fspace_l2_quadrature_field(mesh)
  @show fspace = FunctionSpace(mesh, L2QuadratureField, Lagrange)
end

function test_function_spaces()
  # coords, conns = FiniteElementContainers.create_structured_mesh_data(7, 7, [0., 1.], [0., 1.])
  # target_disp_grad = [
  #   0.1 0.4;
  #   -0.2 -0.1
  # ]
  # coords = H1Field(coords)

  mesh = UnstructuredMesh("mechanics/mechanics.g")
  test_bad_interp_type(mesh)
  test_fspace_h1_field(mesh)
  test_fspace_l2_element_field(mesh)
  test_fspace_l2_quadrature_field(mesh)
end

@testset "Function spaces" begin
  test_function_spaces()
end
