@testset ExtendedTestSet "L2 Element Field" begin
  vals = rand(2, 20)
  field_1 = L2ElementField(vals, (:node_1, :node_2))

  @test size(vals) == size(field_1)

  for n in axes(vals)
    @test vals[n] == field_1[n]
  end
end

@testset ExtendedTestSet "H1 Field" begin
  vals = rand(2, 20)
  field_2 = H1Field(vals, (:u, :v))
  
  @test size(vals) == size(field_2)

  # test basic axes and basic getindex
  for n in axes(vals)
    @test vals[n] == field_2[n]
  end

  # test dual index getindex
  for n in axes(vals, 2)
    for d in axes(vals, 1)
      @test vals[d, n] == field_2[d, n]
    end
  end

  # test dual number setindex
  vals_2 = rand(2, 20)
  for n in axes(vals, 2)
    for d in axes(vals, 1)
      # field_1[d, n] = vals_2[d, n]
      field_2[d, n] = vals_2[d, n]
      # field_3[d, n] = vals_2[d, n]

      # @test vals_2[d, n] == field_1[d, n]
      @test vals_2[d, n] == field_2[d, n]
      # @test vals_2[d, n] == field_3[d, n]
    end
  end

  # test axes with dimension specied
  for n in axes(field_2, 2)
    for d in axes(field_2, 1)
      # @test vals[d, n] == field_1[d, n]
      @test vals[d, n] == field_2[d, n]
      # @test vals[d, n] == field_3[d, n]
    end
  end
end

@testset ExtendedTestSet "Quadrature Field" begin
  vals = rand(Float64, 2, 3, 100)
  field_3 = L2QuadratureField(vals, (:var_1, :var_2))

  @test size(vals) == size(field_3)

  for n in axes(vals)
    @test vals[n] == field_3[n]
  end
end
