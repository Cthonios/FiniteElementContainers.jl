function test_connectivity()
  ref_fe_1 = ReferenceFE(Quad4{Lagrange, 1}())
  ref_fe_2 = ReferenceFE(Tri3{Lagrange, 1}())
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

function test_h1_field()
  data = rand(2, 20)
  field = H1Field(data)
  
  @test eltype(field) == eltype(data)
  @test size(field) == size(data)
  @test num_fields(field) == size(data, 1)
  @test num_nodes(field) == size(data, 2)
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

function test_l2_field()
  a1 = rand(2, 3, 40)
  a2 = rand(3, 4, 10)
  field = L2Field([a1, a2])
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

@testset "Fields" begin
  test_connectivity()
  test_h1_field()
  test_l2_field()
end
