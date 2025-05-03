@testset ExtendedTestSet "L2 Element Field" begin
  vals = rand(2, 20)
  field = L2ElementField(vals, (:node_1, :node_2))

  @test eltype(field) == eltype(vals)
  @test names(field) == (:node_1, :node_2)
  @test size(field) == size(vals)
  @test num_fields(field) == size(vals, 1)
  @test num_nodes_per_element(field) == size(vals, 1)
  @test num_elements(field) == size(vals, 2)

  # test basic axes and basic getindex
  for n in axes(vals)
    @test field[n] == vals[n]
  end

  # test dual index getindex
  for n in axes(vals, 2)
    for d in axes(vals, 1)
      @test field[d, n] == vals[d, n]
    end
  end

  for (n, name) in enumerate(names(field))
    @test field[name] == vals[n, :]
    @test field[name, :] == vals[n, :]
    @test view(field, name) == view(vals, n, :)
    @test view(field, name, :) == view(vals, n, :)
  end

  for e in axes(field, 2)
    for (n, name) in enumerate(names(field))
      @test field[name, e] == vals[n, e]
    end
  end

  # test dual number setindex
  vals_2 = rand(2, 20)
  for n in axes(vals, 2)
    for d in axes(vals, 1)
      field[d, n] = vals_2[d, n]
      @test field[d, n] == vals_2[d, n]
    end
  end
end

@testset ExtendedTestSet "H1 Field" begin
  vals = rand(2, 20)
  field = H1Field(vals, (:u, :v))
  
  @test eltype(field) == eltype(vals)
  @test names(field) == (:u, :v)
  @test size(field) == size(vals)
  @test num_fields(field) == size(vals, 1)
  @test num_nodes(field) == size(vals, 2)

  # test basic axes and basic getindex
  for n in axes(vals)
    @test field[n] == vals[n]
  end

  # test dual index getindex
  for n in axes(vals, 2)
    for d in axes(vals, 1)
      @test field[d, n] == vals[d, n]
    end
  end

  for (n, name) in enumerate(names(field))
    @test field[name] == vals[n, :]
    @test field[name, :] == vals[n, :]
    @test view(field, name) == view(vals, n, :)
    @test view(field, name, :) == view(vals, n, :)
  end

  for e in axes(field, 2)
    for (n, name) in enumerate(names(field))
      @test field[name, e] == vals[n, e]
    end
  end

  # test dual number setindex
  vals_2 = rand(2, 20)
  for n in axes(vals, 2)
    for d in axes(vals, 1)
      field[d, n] = vals_2[d, n]
      @test field[d, n] == vals_2[d, n]
    end
  end
end

@testset ExtendedTestSet "Quadrature Field" begin
  vals = rand(Float64, 2, 3, 100)
  field = L2QuadratureField(vals, (:var_1, :var_2))

  @test eltype(field) == eltype(vals)
  @test names(field) == (:var_1, :var_2)
  @test size(field) == size(vals)
  @test num_elements(field) == size(vals, 3)

  # test basic axes and basic getindex
  for n in axes(vals)
    @test field[n] == vals[n]
  end

  # test dual index getindex
  # for n in axes(vals, 3)
  #   for q in axes(vals, 2)
  #     for e in axes(vals, 1)
  #       @test field[n, q, e] == vals[n, q, e]
  #     end
  #   end
  # end

  for (n, name) in enumerate(names(field))
    @test field[name] == vals[n, :, :]
    @test field[name, :, :] == vals[n, :, :]
    @test view(field, name) == view(vals, n, :, :)
    @test view(field, name, :, :) == view(vals, n, :, :)
  end

  for e in axes(field, 3)
    for q in axes(field, 2)
      for (n, name) in enumerate(names(field))
        @test field[name, q, e] == vals[n, q, e]
      end
    end
  end

  # special case for zero fields
  # NOTE this is the expected behavior below right now
  # even though it is slightly dumb.
  vals = rand(Float64, 0, 3, 100)
  field = L2QuadratureField(vals, ())
  @test size(field) == (0, 3, 0)
  @test axes(field) == (Base.OneTo(0), Base.OneTo(3), Base.OneTo(0))
end
