using Adapt
using AMDGPU
using Aqua
using CUDA
using FiniteElementContainers
using ForwardDiff
using Gmsh
using Krylov
using LinearAlgebra
using PartitionedArrays
using ReferenceFiniteElements
using SparseArrays
using StaticArrays
using Tensors
using Test

# put these first so we can use these
# physics in other tests
include("laplace_with_source/TestLaplaceCommon.jl")
include("mechanics/TestMechanicsCommon.jl")
include("poisson/TestPoissonCommon.jl")
include("poisson/TestPoissonPBCs.jl")

include("TestAssemblers.jl")
include("TestBCs.jl")
include("TestDofManagers.jl")
include("TestFields.jl")
include("TestFormulations.jl")
include("TestFunctions.jl")
include("TestFunctionSpaces.jl")
include("TestICs.jl")
include("TestIntegrals.jl")
include("TestMesh.jl")
include("TestPhysics.jl")

# "Regression" tests below
function test_laplace()
  cg_solver = x -> IterativeLinearSolver(x, :cg)
  condensed = [false, true]
  lsolvers = [cg_solver, DirectLinearSolver]
  # cpu tests
  for cond in condensed
    for lsolver in lsolvers
      test_laplace(cpu, cond, NewtonSolver, lsolver)
    end
  end

  if AMDGPU.functional()
    for cond in condensed
      test_laplace(rocm, cond, NewtonSolver, cg_solver)
    end
  end

  if CUDA.functional()
    for cond in condensed
      test_laplace(cuda, cond, NewtonSolver, cg_solver)
    end
  end
end

function test_poisson()
  cg_solver = x -> IterativeLinearSolver(x, :cg)
  condensed = [false, true]
  lsolvers = [cg_solver, DirectLinearSolver]
  # cpu tests
  for cond in condensed
    for lsolver in lsolvers
      test_poisson(cpu, cond, NewtonSolver, lsolver)
    end
  end
  test_poisson_pbcs()

  if AMDGPU.functional()
    for cond in condensed
      test_poisson(rocm, cond, NewtonSolver, cg_solver)
    end
  end

  if CUDA.functional()
    for cond in condensed
      test_poisson(cuda, cond, NewtonSolver, cg_solver)
    end
  end
end

function test_mechanics()
  cg_solver = x -> IterativeLinearSolver(x, :cg)

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

@testset "Regression tests" begin
  include("laplace_with_source/TestLaplace.jl")
  @testset "Laplace with source" test_laplace()
  include("poisson/TestPoisson.jl")
  @testset "Poisson" test_poisson()
  include("mechanics/TestMechanics.jl")
  @testset "Mechanics" test_mechanics()
end

@testset "Non-mesh Extension tests" begin
  include("ext/TestPartitionedArraysExt.jl")
end

@testset "Aqua" begin
  Aqua.test_all(FiniteElementContainers; ambiguities=false)
end
