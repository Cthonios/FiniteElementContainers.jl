@testitem "Fields - test_connectivity" begin
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
end

@testitem "Fields - test_h1_field" begin
  import KernelAbstractions as KA
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
  a2 = rand(3, 4, 10)
  field = L2Field([a1, a2])
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
