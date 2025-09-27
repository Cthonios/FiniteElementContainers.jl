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
include("TestICs.jl")
include("TestMesh.jl")
include("TestPhysics.jl")

# @testset ExtendedTestSet "Eigen problem" begin
#   include("eigen/TestEigen.jl")
# end

# "Regression" tests below

@testset ExtendedTestSet "Poisson problem" begin
  include("poisson/TestPoisson.jl")

  cg_solver = x -> IterativeLinearSolver(x, :CgSolver)

  # cpu tests
  test_poisson_dirichlet(cpu, false, NewtonSolver, DirectLinearSolver)
  test_poisson_dirichlet(cpu, true, NewtonSolver, DirectLinearSolver)
  test_poisson_dirichlet(cpu, false, NewtonSolver, cg_solver)
  test_poisson_dirichlet(cpu, true, NewtonSolver, cg_solver)
  test_poisson_neumann(cpu, false, NewtonSolver, DirectLinearSolver)
  test_poisson_neumann(cpu, true, NewtonSolver, DirectLinearSolver)
  test_poisson_neumann(cpu, false, NewtonSolver, cg_solver)
  test_poisson_neumann(cpu, true, NewtonSolver, cg_solver)
  
  if AMDGPU.functional()
    test_poisson_dirichlet(rocm, false, NewtonSolver, cg_solver)
    test_poisson_dirichlet(rocm, true, NewtonSolver, cg_solver)
    test_poisson_neumann(rocm, false, NewtonSolver, cg_solver)
    test_poisson_neumann(rocm, true, NewtonSolver, cg_solver)
  end

  if CUDA.functional()
    test_poisson_dirichlet(cuda, false, NewtonSolver, cg_solver)
    test_poisson_dirichlet(cuda, true, NewtonSolver, cg_solver)
    test_poisson_neumann(cuda, false, NewtonSolver, cg_solver)
    test_poisson_neumann(cuda, true, NewtonSolver, cg_solver)
  end
end

@testset ExtendedTestSet "Mechanics Problem" begin
  include("mechanics/TestMechanicsCommon.jl")
  include("mechanics/TestMechanics.jl")

  cg_solver = x -> IterativeLinearSolver(x, :CgSolver)

  test_mechanics_dirichlet_only(cpu, false, NewtonSolver, DirectLinearSolver)
  test_mechanics_dirichlet_only(cpu, true, NewtonSolver, DirectLinearSolver)
  test_mechanics_dirichlet_only(cpu, false, NewtonSolver, cg_solver)
  test_mechanics_dirichlet_only(cpu, true, NewtonSolver, cg_solver)
  
  if AMDGPU.functional()
    test_mechanics_dirichlet_only(rocm, false, NewtonSolver, cg_solver)
    test_mechanics_dirichlet_only(rocm, true, NewtonSolver, cg_solver)
  end

  if CUDA.functional()
    test_mechanics_dirichlet_only(cuda, false, NewtonSolver, cg_solver)
    test_mechanics_dirichlet_only(cuda, true, NewtonSolver, cg_solver)
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
