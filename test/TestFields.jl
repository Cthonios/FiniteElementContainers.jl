@testset ExtendedTestSet "Element Field" begin
  vals = rand(2, 20)

  field_1 = ElementField{2, 20}(vals)
  field_2 = ElementField{2, 20}(vals |> vec)

  @test size(vals) == size(field_1)
  @test size(vals) == size(field_2)

  for n in axes(vals)
    @test vals[n] == field_1[n]
    @test vals[n] == field_2[n]
  end

  # # some constructor tests
  # field = FiniteElementContainers.SimpleElementField{4, 10, Matrix, Float64}(undef)
  # field = zero(typeof(field))
  # field = ElementField{4, 10, Vector, Float64}(undef)
  # field[1, 4] = 4.0
  # @test field[1, 4] ≈ 4.0
  # field = similar(field)
  # field = zero(typeof(field))
  # field = ElementField{4, 10, Matrix, Float64}(undef)
  # field = similar(field)
  # field = zero(field)
  # field = ElementField{(4, 10), Matrix, Float64}(undef)
  # field = ElementField{4, 10, Vector}(vec(field) |> collect)
  # field = ElementField{4, 10, StructArray, SVector{4, Int64}}(undef)
  # field = zero(field)
  # # field = zero(typeof(field))
  # field = FiniteElementContainers.SimpleElementField{4, 10, StructVector, SVector{4, Int64}}(undef)
  # field = zero(field)
  # field = zero(typeof(field))
  # field = FiniteElementContainers.SimpleElementField{4, 10, StructArray, SVector{4, Int64}}(undef)
  # field = zero(typeof(field))
  # # @test size(field) == (10,)
  # field = FiniteElementContainers.VectorizedElementField{4, 4}(
  #   [
  #     SVector{4, Int64}([1, 2, 3, 4]),
  #     SVector{4, Int64}([5, 6, 7, 8]),
  #     SVector{4, Int64}([9, 10, 11, 12]),
  #     SVector{4, Int64}([13, 14, 15, 16])
  #   ]
  # )
  # field = FiniteElementContainers.VectorizedElementField{4, 10, SVector{4, Int64}}(undef)

  # # component array
  # NFS = (1, 2, 3, 4)
  # NES = (10, 100, 1000, 10000)
  # names = (:block_1, :block_2, :block_3, :block_4)
  # field = FiniteElementContainers.ComponentArrayElementField{NFS, NES, Float64}(undef, names)
  # for i in 1:4
  #   @test size(field, i) == (NFS[i], NES[i])
  #   @test size(field[names[i]]) == (NFS[i], NES[i])

  #   temp = rand(Float64, NFS[i], NES[i])
  #   setindex!(field, temp, names[i], :, :)
  #   @test field[names[i]] ≈ temp

  #   # temp = rand(Float64, NFS[i], NES[i])
  #   # # setindex!(field, temp, names[i], :, :)
  #   # field[names[i], :, :] .= field
  #   # @test field[names[i]] ≈ temp
  # end
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
  field_2 = NodalField{2, 20}(vals)
  field_3 = NodalField{2, 20}(vals |> vec)

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
      # field_3[d, n] = vals_2[d, n]

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
  # field = NodalField{1, 10, Vector, Float64}(undef)
  # # @show size(field)
  # # @test size(field) == (10,)
  # field = NodalField{2, 10, Vector, Float64}(undef)
  # field = NodalField{2, 10, Vector}(vec(field) |> collect)
  # field = similar(field)
  # field = zero(field)
  # field = zero(typeof(field))
  # field = FiniteElementContainers.VectorizedNodalField{2, 10, Float64}(undef)
  # field = FiniteElementContainers.VectorizedNodalField{2, 10}(vec(field) |> collect)
  # @test size(field) == (2, 10)
  # field = NodalField{2, 10, Matrix, Float64}(undef)
  # field = NodalField{(2, 10), Matrix, Float64}(undef)
  # field = similar(field)
  # field = zero(field)
  # field = NodalField{2, 10, StructArray, SVector{2, Float64}}(undef)
  # @test size(field) == (10,)

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

# @testset ExtendedTestSet "Quadrature Field" begin
#   vals = rand(Float64, 4, 3, 100)
#   field = QuadratureField{(4,), (3,), (100,), ComponentArray, Float64}(undef, (:block_1,))
#   @test size(field, 1) == (4, 3, 100)
# end