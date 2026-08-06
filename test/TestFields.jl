@testitem "Fields - test_connectivity" begin
  import FiniteElementContainers as FEC
  using ReferenceFiniteElements

  ref_fe_1 = ReferenceFE(Quad{Lagrange, 1}(), GaussLobattoLegendre(1))
  ref_fe_2 = ReferenceFE(Tri{Lagrange, 1}(), GaussLobattoLegendre(1))
  conns_in = [
    [
      1 5 9;
      2 6 10;
      3 7 11;
      4 8 12
    ],
    [
      13 16 19 22 25; 
      14 17 20 23 26;
      15 18 21 24 27
    ]
  ]
  els_in = [
    [1, 2, 3],
    [4, 5, 6, 7, 8]
  ]
  conn = Connectivity(conns_in)
  # testing block view
  block_conn = connectivity(conn, 1)
  @test size(block_conn) == (4, 3)
  @test connectivity(ref_fe_1, conn.data, 1, 1) == [1, 2, 3, 4]
  @test connectivity(ref_fe_1, conn.data, 2, 1) == [5, 6, 7, 8]
  @test connectivity(ref_fe_1, conn.data, 3, 1) == [9, 10, 11, 12]
  block_conn = connectivity(conn, 2)
  @test size(block_conn) == (3, 5)
  @test connectivity(ref_fe_2, conn.data, 1, 13) == [13, 14, 15]
  @test connectivity(ref_fe_2, conn.data, 2, 13) == [16, 17, 18]
  @test connectivity(ref_fe_2, conn.data, 3, 13) == [19, 20, 21]
  @test connectivity(ref_fe_2, conn.data, 4, 13) == [22, 23, 24]
  @test connectivity(ref_fe_2, conn.data, 5, 13) == [25, 26, 27]

  # testing v2 connectivity
  conn = FEC.Connectivity_v2(conn)
  # block 1
  @test map(x -> connectivity(conn, x, 1, 1), 1:4) == [1, 2, 3, 4]
  @test map(x -> connectivity(conn, x, 2, 1), 1:4) == [5, 6, 7, 8]
  @test map(x -> connectivity(conn, x, 3, 1), 1:4) == [9, 10, 11, 12]
  # block 2
  @test map(x -> connectivity(conn, x, 1, 2), 1:3) == [13, 14, 15]
  @test map(x -> connectivity(conn, x, 2, 2), 1:3) == [16, 17, 18]
  @test map(x -> connectivity(conn, x, 3, 2), 1:3) == [19, 20, 21]
  @test map(x -> connectivity(conn, x, 4, 2), 1:3) == [22, 23, 24]
  @test map(x -> connectivity(conn, x, 5, 2), 1:3) == [25, 26, 27]
end

@testitem "Fields - test_h1_field" begin
  import KernelAbstractions as KA
  using Adapt
  if "--test-amdgpu" in ARGS @eval using AMDGPU end
  if "--test-cuda" in ARGS @eval using CUDA end
  include("TestUtils.jl")
  backends = _get_backends()
  data = rand(2, 20)
  field = H1Field(data)
  
  @test eltype(field) == eltype(data)
  @test ndims(field) == 2
  @test size(field) == size(data)
  @test num_fields(field) == size(data, 1)
  @test num_entities(field) == size(data, 2)
  @test typeof(similar(field)) == typeof(field) 
  @test all(unique(field) .≈ unique(field.data))
  @test KA.get_backend(field) == KA.CPU()

  # test adapt
  for backend in backends
    if backend == cpu
      continue
    end
    to = _backend_to_array_type(backend)
    field_gpu = adapt(to, field)
    field_cpu = adapt(Array, field)
    @test all(field_cpu .≈ field)
  end

  # test basic axes and basic getindex
  for n in axes(data)
    @test field[n] == data[n]
  end

  # test dual index getindex
  for n in axes(data, 2)
    for d in axes(data, 1)
      @test field[d, n] == data[d, n]
    end
  end

  # test dual number setindex
  data_2 = rand(2, 20)
  for n in axes(data, 2)
    for d in axes(data, 1)
      field[d, n] = data_2[d, n]
      @test field[d, n] == data_2[d, n]
    end
  end

  # test fill!
  fill!(field, 3.9)
  @test all(field .≈ 3.9)

  # similar
  new_field = similar(field)
  new_field .= field
  @test all(field .≈ new_field)
end

@testitem "Fields - test_hcurl_field" begin
  data = rand(2, 20)
  field = HcurlField(data)
  
  @test eltype(field) == eltype(data)
  @test size(field) == size(data)
  @test num_fields(field) == size(data, 1)
  @test num_entities(field) == size(data, 2)
  @test typeof(similar(field)) == typeof(field) 

  # test basic axes and basic getindex
  for n in axes(data)
    @test field[n] == data[n]
  end

  # test dual index getindex
  for n in axes(data, 2)
    for d in axes(data, 1)
      @test field[d, n] == data[d, n]
    end
  end

  # test dual number setindex
  data_2 = rand(2, 20)
  for n in axes(data, 2)
    for d in axes(data, 1)
      field[d, n] = data_2[d, n]
      @test field[d, n] == data_2[d, n]
    end
  end
end

@testitem "Fields - test_hdiv_field" begin
  data = rand(2, 20)
  field = HdivField(data)
  
  @test eltype(field) == eltype(data)
  @test size(field) == size(data)
  @test num_fields(field) == size(data, 1)
  @test num_entities(field) == size(data, 2)
  @test typeof(similar(field)) == typeof(field) 

  # test basic axes and basic getindex
  for n in axes(data)
    @test field[n] == data[n]
  end

  # test dual index getindex
  for n in axes(data, 2)
    for d in axes(data, 1)
      @test field[d, n] == data[d, n]
    end
  end

  # test dual number setindex
  data_2 = rand(2, 20)
  for n in axes(data, 2)
    for d in axes(data, 1)
      field[d, n] = data_2[d, n]
      @test field[d, n] == data_2[d, n]
    end
  end
end

@testitem "Fields - test_l2_field" begin
  a1 = rand(2, 3, 40)
  a2 = rand(2, 4, 10)
  field = L2Field([a1, a2])
  @show field
  @test size(FiniteElementContainers.block_view(field, 1)) == (2, 3, 40)
  @test size(FiniteElementContainers.block_view(field, 2)) == (2, 4, 10)

  bview = FiniteElementContainers.block_view(field, 1)
  for k in axes(bview, 3)
    for j in axes(bview, 2)
      for i in axes(bview, 1)
        @test bview[i, j, k] ≈ a1[i, j, k]
      end
    end
  end

  bview = FiniteElementContainers.block_view(field, 2)
  for k in axes(bview, 3)
    for j in axes(bview, 2)
      for i in axes(bview, 1)
        @test bview[i, j, k] ≈ a2[i, j, k]
      end
    end
  end
end

@testitem "Fields - test_property_field_all_constant" begin
  props_1 = rand(2)
  props_2 = rand(3)
  props = FiniteElementContainers.PropertyField([props_1, props_2])
  @test all(FiniteElementContainers.properties(props, 1, 1) .≈ props_1)
  @test all(FiniteElementContainers.properties(props, 100, 1) .≈ props_1)

  @test all(FiniteElementContainers.properties(props, 1, 2) .≈ props_2)
  @test all(FiniteElementContainers.properties(props, 100, 2) .≈ props_2)
end

@testitem "Fields - test_property_field_,mixed_constant_and_element_level" begin
  props_1 = rand(3)
  props_2 = rand(4, 20)
  props = FiniteElementContainers.PropertyField([props_1, props_2])
  @test all(FiniteElementContainers.properties(props, 1, 1) .≈ props_1)
  @test all(FiniteElementContainers.properties(props, 100, 1) .≈ props_1)

  for e in axes(props_2, 2)
    @test all(FiniteElementContainers.properties(props, e, 2) .≈ props_2[:, e])
  end
end

@testitem "Fields - test_property_field_all_element_level" begin
  props_1 = rand(3, 10)
  props_2 = rand(4, 20)
  props = FiniteElementContainers.PropertyField([props_1, props_2])
  for e in axes(props_1, 2)
    @test all(FiniteElementContainers.properties(props, e, 1) .≈ props_1[:, e])
  end
  for e in axes(props_2, 2)
    @test all(FiniteElementContainers.properties(props, e, 2) .≈ props_2[:, e])
  end
end

@testitem "Fields - test_state_variable_field" begin
  a1 = rand(2, 3, 40)
  a2 = rand(3, 4, 10)
  field = StateVariableField([a1, a2])
  @show field
  @test size(FiniteElementContainers.block_view(field, 1)) == (2, 3, 40)
  @test size(FiniteElementContainers.block_view(field, 2)) == (3, 4, 10)

  bview = FiniteElementContainers.block_view(field, 1)
  for k in axes(bview, 3)
    for j in axes(bview, 2)
      for i in axes(bview, 1)
        @test bview[i, j, k] ≈ a1[i, j, k]
      end
    end
  end

  bview = FiniteElementContainers.block_view(field, 2)
  for k in axes(bview, 3)
    for j in axes(bview, 2)
      for i in axes(bview, 1)
        @test bview[i, j, k] ≈ a2[i, j, k]
      end
    end
  end
end