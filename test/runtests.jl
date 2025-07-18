using Adapt
using AMDGPU
using Aqua
using CUDA
using Exodus
using FiniteElementContainers
using ForwardDiff
# using JET
using LinearAlgebra
using ReferenceFiniteElements
using StaticArrays
using Tensors
using Test
using TestSetExtensions

include("TestAssemblers.jl")
include("TestBCs.jl")
include("TestConnectivities.jl")
include("TestDofManagers.jl")
include("TestFields.jl")
include("TestFormulations.jl")
include("TestFunctions.jl")
include("TestFunctionSpaces.jl")
include("TestMesh.jl")
include("TestPhysics.jl")

# @testset ExtendedTestSet "Eigen problem" begin
#   include("eigen/TestEigen.jl")
# end

@testset ExtendedTestSet "Poisson problem" begin
  include("poisson/TestPoisson.jl")
  if AMDGPU.functional()
    include("poisson/TestPoissonAMDGPU.jl")
  end
  if CUDA.functional()
    include("poisson/TestPoissonCUDA.jl")
  end
end

@testset ExtendedTestSet "Mechanics Problem" begin
  include("mechanics/TestMechanicsCommon.jl")
  include("mechanics/TestMechanics.jl")
  if AMDGPU.functional()
    include("mechanics/TestMechanicsAMDGPU.jl")
  end
  if CUDA.functional()
    include("mechanics/TestMechanicsCUDA.jl")
  end
end

@testset ExtendedTestSet "Aqua" begin
  Aqua.test_all(FiniteElementContainers; ambiguities=false)
end

# getting an error from FileMesh.num_nodes for some reason. No method
# @testset ExtendedTestSet "JET" begin
#   JET.test_package(FiniteElementContainers; target_defined_modules=true)
#   # JET.test_package(FiniteElementContainers)
# end
