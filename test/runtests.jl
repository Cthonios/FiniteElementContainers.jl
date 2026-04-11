using Adapt
if "--test-amdgpu" in ARGS @eval using AMDGPU end
using Aqua
if "--test-cuda" in ARGS @eval using CUDA end
using Exodus
using FiniteElementContainers
using ForwardDiff
using Gmsh
using Krylov
using LinearAlgebra
# using PartitionedArrays
using ReferenceFiniteElements
using SparseArrays
using SparseMatricesCSR
using StaticArrays
using Tensors
using Test

# some helpers
function _check_functional_backend(name)
  if isdefined(Main, name)
    return eval(name).functional()
  else
    return false
  end
end

function _get_backends()
  if "--ignore-cpu" in ARGS
    backends = Function[]
  else 
    backends = Function[cpu]
  end
  if _check_functional_backend(:AMDGPU)
    push!(backends, rocm)
  end
  if _check_functional_backend(:CUDA)
    push!(backends, cuda)
  end
  return backends
end

# "Regression" tests below

# laplace mainly covers the presence of sources
# we don't need to cover each and every combinatoric of options
function test_laplace()
  backends = _get_backends()
  cg_solver = x -> IterativeLinearSolver(x, :cg)
  use_condensed = [false, true]

  for backend in backends
    for cond in use_condensed
      @info "Test Laplace with $backend, $cg_solver, use condensed = $cond"
      test_laplace(
        backend, NewtonSolver, cg_solver;
        sparse_matrix_type = :csc,
        use_condensed = cond,
        use_inplace_methods = false
      )
    end
  end
end

function test_poisson()
  backends = _get_backends()
  use_condensed = [false, true]
  use_inplace_methods = [false, true]

  for backend in backends
    if backend == cpu
      lsolver = DirectLinearSolver
    else
      lsolver = x -> IterativeLinearSolver(x, :cg)
    end
    for cond in use_condensed
      for use_inplace_method in use_inplace_methods
        @info "Test Poisson with $backend $cond $use_inplace_method"
        test_poisson(
          backend, NewtonSolver, lsolver;
          sparse_matrix_type = :csc,
          use_condensed = cond,
          use_inplace_methods = use_inplace_method
        )
      end
    end
  end

  # TODO move this to first class status
  test_poisson_pbcs()
end

function test_mechanics()
  backends = _get_backends()
  cg_solver = x -> IterativeLinearSolver(x, :cg)
  sparse_matrix_types = [:csc, :csr]
  use_condensed = [false, true]
  use_inplace_methods = [false, true]

  for backend in backends
    for cond in use_condensed
      for sparse_matrix_type in sparse_matrix_types
        for use_inplace_method in use_inplace_methods
          if backend != cpu
            lsolver = cg_solver
          else
            if sparse_matrix_type == :csc
              lsolver = DirectLinearSolver
            elseif sparse_matrix_type == :csr
              lsolver = cg_solver
            end
          end
          @info "Test mechanics $backend, $cond, $sparse_matrix_type, $use_inplace_method, $lsolver"
          test_mechanics_dirichlet_only(
            backend, NewtonSolver, lsolver;
            sparse_matrix_type = sparse_matrix_type,
            use_condensed = cond, use_inplace_methods = use_inplace_method
          )
          # end
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
