@testset ExtendedTestSet "Element Field" begin
  vals = rand(2, 20)

  field_1 = ElementField{2, 20, Matrix}(vals)
  field_2 = ElementField{2, 20, Vector}(vals)

  @test size(vals) == size(field_1)
  @test size(vals) == size(field_2)

  for n in axes(vals)
    @test vals[n] == field_1[n]
    @test vals[n] == field_2[n]
  end
end

@testset ExtendedTestSet "Nodal Field" begin
  # vals = rand(2, 20)
  # field = NodalField{2, 20}(vals, :vals)
  # @test size(vals) == size(field)
  
  # for n in axes(vals)
  #   @test vals[n] == field[n]
  # end

  # test regular constructors
  vals = rand(2, 20)

  # field_1 = NodalField{2, 20}(vals)
  field_2 = NodalField{2, 20, Matrix}(vals)
  field_3 = NodalField{2, 20, Vector}(vals)

  # field_2 .= vec(vals)
  # field_2 .= vals
  # field_3 .= vals

  # test basic axes and basic getindex
  for n in axes(vals)
    # @test vals[n] == field_1[n]
    @test vals[n] == field_2[n]
    @test vals[n] == field_3[n]
  end

  # test dual index getindex
  for n in axes(vals, 2)
    for d in axes(vals, 1)
      # @test vals[d, n] == field_1[d, n]
      @test vals[d, n] == field_2[d, n]
      @test vals[d, n] == field_3[d, n]
    end
  end

  # test dual number setindex
  vals_2 = rand(2, 20)
  for n in axes(vals, 2)
    for d in axes(vals, 1)
      # field_1[d, n] = vals_2[d, n]
      field_2[d, n] = vals_2[d, n]
      field_3[d, n] = vals_2[d, n]

      # @test vals_2[d, n] == field_1[d, n]
      @test vals_2[d, n] == field_2[d, n]
      @test vals_2[d, n] == field_3[d, n]
    end
  end

  # test axes with dimension specied
  for n in axes(field_2, 2)
    for d in axes(field_2, 1)
      # @test vals[d, n] == field_1[d, n]
      @test vals[d, n] == field_2[d, n]
      @test vals[d, n] == field_3[d, n]
    end
  end

  # some constructor tests
  field = NodalField{2, 10, Vector, Float64}(undef)
  field = NodalField{2, 10, Vector}(vec(field))
  field = similar(field)
  field = zero(field)
  field = NodalField{2, 10, Matrix, Float64}(undef)
  field = NodalField{(2, 10), Matrix, Float64}(undef)
  field = similar(field)
  field = zero(field)
  field = NodalField{2, 10, StructArray, SVector{2, Float64}}(undef)

  # # some constructor tests
  # field = FiniteElementContainers.SimpleNodalField{2, 10, Float64}(undef)
  # field = similar(field)
  # field = zero(field)
  # @test all(x -> x ≈ 0.0, field)

  # # some constructor tests
  # field = FiniteElementContainers.VectorizedNodalField{2, 10, Float64}(undef)
  # field = similar(field)
  # field = zero(field)
  # @test all(x -> x ≈ 0.0, field)
  # field = FiniteElementContainers.VectorizedNodalField{2, 10}(vec(field))
  # field = FiniteElementContainers.VectorizedNodalField{2, 10, StructArray, SVector{2, Float64}}(undef)
  # field = zero(field)
end
