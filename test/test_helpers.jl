# local usings
using Exodus
using FiniteElementContainers
using ForwardDiff
using IterativeSolvers
using LinearAlgebra
using PaddedViews
using Parameters
using Printf
using StaticArrays

# Regression testing helper methods
function read_mesh(file_name, nsets)
  return Mesh(file_name; nsets=nsets)
end

function container_setup(mesh, block_ids, q_degree, ndofs; is_dynamic=false)
  fspaces = FunctionSpace[]
  for block_id in block_ids
    push!(fspaces, FunctionSpace(mesh, block_id, q_degree, ndofs))
  end
  dof = DofManager(mesh, ndofs)
  if is_dynamic
    asm = DynamicAssembler(dof, fspaces)
  else
    asm = StaticAssembler(dof, fspaces)
  end
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

function simple_eigen_solver(mesh, fspaces, dof, asm, bcs, residual, tangent, mass)
  U = create_fields(dof)
  update_bcs!(U, mesh, dof, bcs)
  Uu = create_unknowns(dof)

  update_fields!(U, dof, Uu)
  assemble!(asm, fspaces, dof, residual, tangent, mass, U)
  results = lobpcg(asm.K, asm.M, false, 100)
  return results.λ, results.X
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

function simple_eigen_post_processor(mesh_file, eig_vals, eig_vecs, var_name)
  exo_file = splitext(mesh_file)[1] * ".e"
  copy_mesh(mesh_file, exo_file)
  exo = ExodusDatabase(exo_file, "rw")
  write_names(exo, NodalVariable, [var_name])
  for (n, eig_val) in enumerate(eig_vals)
    write_time(exo, n, eig_val)
    write_values(exo, NodalVariable, n, var_name, eig_vecs[:, n])
  end
  close(exo)
end
