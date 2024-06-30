using Aqua
using Exodus
using FiniteElementContainers
using JET
using LinearAlgebra
using Parameters
using ReferenceFiniteElements
using StaticArrays
using StructArrays
using Test
using TestSetExtensions

include("TestConnectivities.jl")
include("TestDofManagers.jl")
include("TestFields.jl")
include("TestFunctionSpaces.jl")
include("TestMesh.jl")

# @testset ExtendedTestSet "Eigen problem" begin
#   include("eigen/TestEigen.jl")
# end

@testset ExtendedTestSet "Poisson problem" begin
  include("poisson/TestPoisson.jl")
end

@testset ExtendedTestSet "Aqua" begin
  Aqua.test_all(FiniteElementContainers; ambiguities=false)
end

# getting an error from FileMesh.num_nodes for some reason. No method
# @testset ExtendedTestSet "JET" begin
#   JET.test_package(FiniteElementContainers; target_defined_modules=true)
# end
