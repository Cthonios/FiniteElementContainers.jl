using Exodus
using FiniteElementContainers
using Tensors

# mesh file
gold_file = "./test/mechanics/mechanics.gold"
mesh_file = "./test/mechanics/mechanics.g"
output_file = "./test/mechanics/mechanics.e"

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

# function mechanics_test()
  mesh = UnstructuredMesh(mesh_file)
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = Mechanics(PlaneStrain())

  u = VectorFunction(V, :displ)
  asm = SparseMatrixAssembler(H1Field, u)

  dbcs = DirichletBC[
    DirichletBC(:displ_x, :sset_3, fixed),
    DirichletBC(:displ_y, :sset_3, fixed),
    DirichletBC(:displ_x, :sset_1, fixed),
    DirichletBC(:displ_y, :sset_1, displace),
  ]

  # pre-setup some scratch arrays
  p = create_parameters(asm, physics, dbcs)

  U = create_field(asm, H1Field)
  @time assemble!(asm, physics, U, :stiffness)
  K = stiffness(asm)

  # move to device
  p_gpu = p |> gpu
  asm_gpu = asm |> gpu

  solver = NewtonSolver(IterativeSolver(asm_gpu, :CgSolver))
  Uu = create_unknowns(asm_gpu)

  update_bcs!(H1Field, solver, Uu, p_gpu)

  @time FiniteElementContainers.solve!(solver, Uu, p_gpu)
  @time FiniteElementContainers.solve!(solver, Uu, p_gpu)

  p = p_gpu |> cpu

  pp = PostProcessor(mesh, output_file, u)
  write_times(pp, 1, 0.0)
  write_field(pp, 1, p.h1_field)
  close(pp)
# end

# mechanics_test()
