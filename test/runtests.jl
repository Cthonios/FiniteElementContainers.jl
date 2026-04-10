using Adapt
using AMDGPU
using Aqua
using CUDA
using Exodus
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

# some helpers
function _get_backends()
  backends = Function[cpu]
  if AMDGPU.functional()
    push!(backends, rocm)
  end
  if CUDA.functional()
    push!(backends, cuda)
  end
  return backends
end

# "Regression" tests below
function test_laplace()
  backends = _get_backends()
  cg_solver = x -> IterativeLinearSolver(x, :cg)
  lsolvers = [cg_solver, DirectLinearSolver]
  use_condensed = [false, true]
  use_inplace_methods = [false, true]

  for backend in backends
    for cond in use_condensed
      for use_inplace_method in use_inplace_methods
        for lsolver in lsolvers
          if backend != cpu && lsolver == DirectLinearSolver
            continue
          end
          test_laplace(backend, NewtonSolver, lsolver; use_condensed = cond, use_inplace_methods = use_inplace_method)
        end
      end
    end
  end
end

function test_poisson()
  backends = _get_backends()
  cg_solver = x -> IterativeLinearSolver(x, :cg)
  lsolvers = [cg_solver, DirectLinearSolver]
  use_condensed = [false, true]
  use_inplace_methods = [false, true]

  for backend in backends
    for cond in use_condensed
      for use_inplace_method in use_inplace_methods
        for lsolver in lsolvers
          if backend != cpu && lsolver == DirectLinearSolver
            continue
          end
          test_poisson(backend, NewtonSolver, lsolver; use_condensed = cond, use_inplace_methods = use_inplace_method)
        end
      end
    end
  end

  # TODO move this to first class status
  test_poisson_pbcs()
end

function test_mechanics()
  backends = _get_backends()
  cg_solver = x -> IterativeLinearSolver(x, :cg)
  lsolvers = [cg_solver, DirectLinearSolver]
  lsolvers = [DirectLinearSolver]
  use_condensed = [false, true]
  use_inplace_methods = [false, true]

  for backend in backends
    for cond in use_condensed
      for use_inplace_method in use_inplace_methods
        for lsolver in lsolvers
          if backend != cpu && lsolver == DirectLinearSolver
            continue
          end
          test_mechanics_dirichlet_only(backend, NewtonSolver, lsolver; use_condensed = cond, use_inplace_methods = use_inplace_method)
        end
      end
    end
  end
end

# put these first so we can use these
# physics in other tests
include("laplace_with_source/TestLaplaceCommon.jl")
include("mechanics/TestMechanicsCommon.jl")
include("poisson/TestPoissonCommon.jl")
include("poisson/TestPoissonPBCs.jl")

@testset "Unit tests" begin
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
  Aqua.test_all(FiniteElementContainers)
end
