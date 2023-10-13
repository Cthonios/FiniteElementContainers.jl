using Exodus
using FiniteElementContainers
using LinearAlgebra
using Parameters
using Printf
using StaticArrays

f(X, _) = 2. * π^2 * sin(π * X[1]) * sin(π * X[2])

function residual(cell, u_el)
  @unpack X, N, ∇N_X, JxW = cell
	# X, N, ∇N_X, JxW = cell
  ∇u_q = ∇N_X' * u_el[1, :]
  R_q = (∇N_X * ∇u_q)' .- N' * f(X, 0.0)
  return JxW * R_q[:]
end

function tangent(cell, _)
  @unpack X, N, ∇N_X, JxW = cell
	# X, N, ∇N_X, JxW = cell
  K_q = ∇N_X * ∇N_X'
  return JxW * K_q
end

function read_mesh()
	mesh = Mesh("mesh.g"; nsets=[1, 2, 3, 4])
end

function setup(mesh::Mesh{F, I, B}, bcs, q_degree) where {F, I, B}
	fspace = FunctionSpace(mesh, 1, q_degree)
  dof    = DofManager(mesh, 1)
	asm    = StaticAssembler(dof)

	return fspace, dof, asm
end

function solve(mesh, fspace, dof, asm, bcs)
	U = create_fields(dof)
	update_bcs!(U, mesh, dof, bcs)
	Uu = create_unknowns(dof)

	assemble!(asm, fspace, dof, residual, tangent, U)

	for n in 1:10
		update_fields!(U, dof, Uu)
		# assemble!(asm, fspace, dof, residual, tangent, U)

		if n > 1
			assemble!(asm, fspace, dof, residual, U)
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

function post_process(U)
	copy_mesh("mesh.g", "poisson_output.e")
  exo = ExodusDatabase("poisson_output.e", "rw")
  write_names(exo, NodalVariable, ["u"])
  write_time(exo, 1, 0.0)
  write_values(exo, NodalVariable, 1, "u", U[1, :])
  close(exo)
end

function run_simulation()
	q_degree = 2

	# mesh = Mesh("mesh.g"; nsets=[1, 2, 3, 4])
	mesh = read_mesh()

	bcs = EssentialBC[
		EssentialBC(mesh, 1, 1)
		EssentialBC(mesh, 2, 1)
		EssentialBC(mesh, 3, 1)
		EssentialBC(mesh, 4, 1)
	]

	fspace, dof, asm = setup(mesh, bcs, q_degree)
	U = solve(mesh, [fspace], dof, asm, bcs)
	post_process(U)
end