using Aqua
using JET
using Test
using TestSetExtensions

# seperating so it's easier to set up a test
# outside of runtests.jl
include("test_helpers.jl")

# Regression testing
@testset ExtendedTestSet "Regression Tests" begin
  regression_test_dirs = filter(isdir, readdir("./"))

  jl_files = String[]
  for regression_test in regression_test_dirs
    jl_file = filter(x -> splitext(x)[2] == ".jl", readdir(regression_test))
    # @assert length(jl_file) == 1
    if length(jl_file) != 1
      @warn "Missing regression test for folder $regression_test"
      continue
    end
    push!(jl_files, joinpath(regression_test, jl_file[1]))
  end

  for jl_file in jl_files
    println("\nRunning regression test from $(jl_file)...")
    include(jl_file)
  end
end

# Aqua testing

@testset ExtendedTestSet "Aqua" begin
  Aqua.test_ambiguities(FiniteElementContainers)
  Aqua.test_unbound_args(FiniteElementContainers)
  Aqua.test_undefined_exports(FiniteElementContainers)
  Aqua.test_piracy(FiniteElementContainers)
  Aqua.test_project_extras(FiniteElementContainers)
  Aqua.test_stale_deps(FiniteElementContainers)
  Aqua.test_deps_compat(FiniteElementContainers)
  Aqua.test_project_toml_formatting(FiniteElementContainers)
end


# JET Testing
@testset ExtendedTestSet "JET" begin
  JET.report_package(FiniteElementContainers)
  # JET.test_package(FiniteElementContainers)
end