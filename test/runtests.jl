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
  vals = rand(2, 20)
  field = NodalField{2, 20}(vals, :vals)
  @test size(vals) == size(field)
  
  for n in axes(vals)
    @test vals[n] == field[n]
  end
end

@testset ExtendedTestSet "Poisson problem" begin
  include("TestPoisson.jl")
end
