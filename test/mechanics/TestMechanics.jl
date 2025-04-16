using Exodus
using FiniteElementContainers
using Tensors

# mesh file
gold_file = "./mechanics/mechanics.gold"
mesh_file = "./mechanics/mechanics.g"
output_file = "./mechanics/mechanics.e"

fixed(_, _) = 0.
displace(_, _) = 1.e-3

struct Mechanics{Form} <: AbstractPhysics{2, 0, 0}
  formulation::Form
end

function strain_energy(∇u)
  K = 10.e9
  G = 1.e9
  ε = symmetric(∇u)
  ψ = 0.5 * K * tr(ε)^2 + G * dcontract(dev(ε), dev(ε))
end

function FiniteElementContainers.residual(physics::Mechanics, cell, u_el, args...)
  (; X_q, N, ∇N_X, JxW) = cell

  # kinematics
  ∇u_q = u_el * ∇N_X
  ∇u_q = modify_field_gradients(physics.formulation, ∇u_q)
  # constitutive
  P_q = gradient(strain_energy, ∇u_q)
  # turn into voigt notation
  P_q = extract_stress(physics.formulation, P_q)
  G_q = discrete_gradient(physics.formulation, ∇N_X)
  f_q = G_q * P_q
  return JxW * f_q[:]
  # return f_q
end

function FiniteElementContainers.stiffness(physics::Mechanics, cell, u_el, args...)
  (; X_q, N, ∇N_X, JxW) = cell

  # kinematics
  ∇u_q = u_el * ∇N_X
  ∇u_q = modify_field_gradients(physics.formulation, ∇u_q)
  # constitutive
  K_q = hessian(z -> strain_energy(z), ∇u_q)
  # turn into voigt notation
  K_q = extract_stiffness(physics.formulation, K_q)
  G_q = discrete_gradient(physics.formulation, ∇N_X)
  return JxW * G_q * K_q * G_q'
end

function mechanics_test()
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Mechanics(PlaneStrain())

  u = VectorFunction(V, :displ)
  asm = SparseMatrixAssembler(H1Field, u)

  dbcs = DirichletBC[
    DirichletBC(asm.dof, :displ_x, :sset_3, fixed),
    DirichletBC(asm.dof, :displ_y, :sset_3, fixed),
    DirichletBC(asm.dof, :displ_x, :sset_1, fixed),
    DirichletBC(asm.dof, :displ_y, :sset_1, displace),
  ]
  # update_dofs!(asm, dbcs)

  # pre-setup some scratch arrays
  # Uu = create_unknowns(asm)
  p = create_parameters(asm, physics, dbcs)
  update_dofs!(asm, p)
  Uu = create_unknowns(asm)

  solver = NewtonSolver(IterativeSolver(asm, :CgSolver))
  update_bcs!(H1Field, solver, Uu, p)

  FiniteElementContainers.solve!(solver, Uu, p)

  pp = PostProcessor(mesh, output_file, u)
  write_times(pp, 1, 0.0)
  write_field(pp, 1, p.h1_field)
  # write_field(pp, 1, U)
  close(pp)
end

mechanics_test()
