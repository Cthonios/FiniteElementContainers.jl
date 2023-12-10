using Aqua
using Exodus
using FiniteElementContainers
using JET
using LinearAlgebra
using Parameters
using Test
using TestSetExtensions

@testset ExtendedTestSet "Exodus Mesh Read" begin
  mesh = Mesh(ExodusDatabase, "./mesh.g")
end

@testset ExtendedTestSet "Element Field" begin
  vals = rand(2, 20)
  field = ElementField{2, 20}(vals, :vals)
  @test size(vals) == size(field)
  
  for n in axes(vals)
    @test vals[n] == field[n]
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

  field_1 = NodalField{2, 20}(vals, :vals)
  field_2 = NodalField{2, 20, Vector, Float64}(undef, :vals)
  field_3 = NodalField{2, 20, Matrix, Float64}(undef, :vals)

  field_2 .= vec(vals)
  field_3 .= vals

  # test basic axes and basic getindex
  for n in axes(vals)
    @test vals[n] == field_1[n]
    @test vals[n] == field_2[n]
    @test vals[n] == field_3[n]
  end

  # test dual index getindex
  for n in axes(vals, 2)
    for d in axes(vals, 1)
      @test vals[d, n] == field_1[d, n]
      @test vals[d, n] == field_2[d, n]
      @test vals[d, n] == field_3[d, n]
    end
  end

  # test dual number setindex
  vals_2 = rand(2, 20)
  for n in axes(vals, 2)
    for d in axes(vals, 1)
      field_1[d, n] = vals_2[d, n]
      field_2[d, n] = vals_2[d, n]
      field_3[d, n] = vals_2[d, n]

      @test vals_2[d, n] == field_1[d, n]
      @test vals_2[d, n] == field_2[d, n]
      @test vals_2[d, n] == field_3[d, n]
    end
  end

  # test axes with dimension specied
  for n in axes(field_2, 2)
    for d in axes(field_2, 1)
      @test vals[d, n] == field_1[d, n]
      @test vals[d, n] == field_2[d, n]
      @test vals[d, n] == field_3[d, n]
    end
  end
end

# @testset ExtendedTestSet "Poisson problem" begin
#   include("TestPoisson.jl")
# end

# @testset ExtendedTestSet "Aqua" begin
#   Aqua.test_all(FiniteElementContainers; ambiguities=false)
# end

# @testset ExtendedTestSet "JET" begin
#   JET.test_package(FiniteElementContainers)
# end
