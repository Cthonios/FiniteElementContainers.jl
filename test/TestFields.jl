@testset ExtendedTestSet "L2 Element Field" begin
  data = rand(2, 20)
  field = L2ElementField(data)

  @test eltype(field) == eltype(data)
  @test size(field) == size(data)
  @test num_fields(field) == size(data, 1)
  @test num_nodes_per_element(field) == size(data, 1)
  @test num_elements(field) == size(data, 2)
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

@testset ExtendedTestSet "H1 Field" begin
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

@testset ExtendedTestSet "Quadrature Field" begin
  data = rand(Float64, 2, 3, 100)
  field = L2QuadratureField(data)

  @test eltype(field) == eltype(data)
  @test size(field) == size(data)
  @test num_elements(field) == size(data, 3)
  @test typeof(similar(field)) == typeof(field) 

  # test basic axes and basic getindex
  for n in axes(data)
    @test field[n] == data[n]
  end

  # test dual index getindex
  # for n in axes(data, 3)
  #   for q in axes(data, 2)
  #     for e in axes(data, 1)
  #       @test field[n, q, e] == data[n, q, e]
  #     end
  #   end
  # end

  # special case for zero fields
  # NOTE this is the expected behavior below right now
  # even though it is slightly dumb.
  data = rand(Float64, 0, 3, 100)
  field = L2QuadratureField(data)
  @test size(field) == (0, 3, 0)
  @test axes(field) == (Base.OneTo(0), Base.OneTo(3), Base.OneTo(0))
end
