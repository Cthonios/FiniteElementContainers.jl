using Aqua
using Exodus
using FiniteElementContainers
using ForwardDiff
using JET
using LinearAlgebra
using Parameters
using Printf
using Test
using TestSetExtensions

# Regression testing helper methods
function read_mesh(file_name, nsets)
  return Mesh(file_name; nsets=nsets)
end

function container_setup(mesh, block_ids, q_degree, ndofs)
  fspaces = []
  for block_id in block_ids
    push!(fspaces, FunctionSpace(mesh, block_id, q_degree))
  end
  dof    = DofManager(mesh, ndofs)
  asm    = StaticAssembler(dof)
  return fspaces, dof, asm
end

function simple_solver(mesh, fspaces, dof, asm, bcs, residual, tangent)
  U = create_fields(dof)
    update_bcs!(U, mesh, dof, bcs)
    Uu = create_unknowns(dof)

    assemble!(asm, fspaces, dof, residual, tangent, U)

    for n in 1:10
      update_fields!(U, dof, Uu)

      if n > 1
        assemble!(asm, fspaces, dof, residual, U)
      end
      
      R = asm.R[dof.unknown_indices]
      K = asm.K[dof.unknown_indices, dof.unknown_indices]
      
      # inefficient linear solve
      ΔUu = -K \ R

      @printf "|R|   = %1.6e  |ΔUu| = %1.6e\n" norm(R) norm(ΔUu)

      if (norm(R) < 1e-12) || (norm(ΔUu) < 1e-12)
        break
      end
      Uu = Uu + ΔUu
    end

    update_fields!(U, dof, Uu)
    return U
end

function simple_post_processor(mesh_file, U, var_names)
  @assert size(U, 1) == length(var_names)
  exo_file = splitext(mesh_file)[1] * ".e"
  copy_mesh(mesh_file, exo_file)
  exo = ExodusDatabase(exo_file, "rw")
  write_names(exo, NodalVariable, var_names)
  write_time(exo, 1, 0.0)
  for (n, var_name) in enumerate(var_names)
    write_values(exo, NodalVariable, 1, var_name, U[n, :])
  end
  close(exo)
end

# Regression testing
@testset ExtendedTestSet "Regression Tests" begin
  regression_test_dirs = filter(isdir, readdir("./"))

  jl_files = String[]
  for regression_test in regression_test_dirs
    jl_file = filter(x -> splitext(x)[2] == ".jl", readdir(regression_test))
    @assert length(jl_file) == 1
    println("Running regression test from $(joinpath(regression_test, jl_file[1]))...")
    push!(jl_files, joinpath(regression_test, jl_file[1]))
  end

  for jl_file in jl_files
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