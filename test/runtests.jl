using Adapt
using AMDGPU
using Aqua
using CUDA
using Exodus
using FiniteElementContainers
using ForwardDiff
using LinearAlgebra
using MPI
using ReferenceFiniteElements
using StaticArrays
using Tensors
using Test
# using TestSetExtensions

include("TestAssemblers.jl")
include("TestBCs.jl")
include("TestConnectivities.jl")
include("TestDofManagers.jl")
include("TestFields.jl")
include("TestFormulations.jl")
include("TestFunctions.jl")
include("TestFunctionSpaces.jl")
include("TestICs.jl")
include("TestIntegrals.jl")
include("TestMesh.jl")
include("TestPhysics.jl")

# @testset ExtendedTestSet "Eigen problem" begin
#   include("eigen/TestEigen.jl")
# end

# "Regression" tests below

@testset "Poisson problem" begin
  include("poisson/TestPoisson.jl")

  cg_solver = x -> IterativeLinearSolver(x, :CgSolver)
  condensed = [false, true]
  lsolvers = [cg_solver, DirectLinearSolver]
  # cpu tests
  for cond in condensed
    for lsolver in lsolvers
      test_poisson_dirichlet(cpu, cond, NewtonSolver, lsolver)
      test_poisson_dirichlet_multi_block_quad4_quad4(cpu, cond, NewtonSolver, lsolver)
      test_poisson_dirichlet_multi_block_quad4_tri3(cpu, cond, NewtonSolver, lsolver)
      test_poisson_neumann(cpu, cond, NewtonSolver, lsolver)
    end
  end

  if AMDGPU.functional()
    for cond in condensed
      test_poisson_dirichlet(rocm, cond, NewtonSolver, cg_solver)
      test_poisson_dirichlet_multi_block_quad4_quad4(rocm, cond, NewtonSolver, cg_solver)
      test_poisson_dirichlet_multi_block_quad4_tri3(rocm, cond, NewtonSolver, cg_solver)
      test_poisson_neumann(rocm, cond, NewtonSolver, cg_solver)
    end
  end

  if CUDA.functional()
    for cond in condensed
      test_poisson_dirichlet(cuda, cond, NewtonSolver, cg_solver)
      test_poisson_dirichlet_multi_block_quad4_quad4(cuda, cond, NewtonSolver, cg_solver)
      test_poisson_dirichlet_multi_block_quad4_tri3(cuda, cond, NewtonSolver, cg_solver)
      test_poisson_neumann(cuda, cond, NewtonSolver, cg_solver)
    end
  end
end

@testset "Mechanics Problem" begin
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

@testset "Aqua" begin
  Aqua.test_all(FiniteElementContainers; ambiguities=false)
end

# getting an error from FileMesh.num_nodes for some reason. No method
# @testset ExtendedTestSet "JET" begin
#   JET.test_package(FiniteElementContainers; target_defined_modules=true)
#   # JET.test_package(FiniteElementContainers)
# end
