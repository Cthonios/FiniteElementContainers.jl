using Exodus
using FiniteElementContainers
using StaticArrays
using Tensors

# mesh file
gold_file = "./mechanics/mechanics.gold"
mesh_file = "./mechanics/mechanics.g"
output_file = "./mechanics/mechanics.e"

fixed(_, _) = 0.
displace(_, t) = 1.e-3 * t

struct Mechanics{Form} <: AbstractPhysics{2, 2, 0}
  formulation::Form
end

function FiniteElementContainers.create_properties(::Mechanics)
  K = 10.e9
  G = 1.e9
  return SVector{2, Float64}(K, G)
end

function strain_energy(∇u)
  K = 10.e9
  G = 1.e9
  ε = symmetric(∇u)
  ψ = 0.5 * K * tr(ε)^2 + G * dcontract(dev(ε), dev(ε))
end

function FiniteElementContainers.residual(
  physics::Mechanics, interps, u_el, x_el, state_old_q, props_el, dt
)
  (; X_q, N, ∇N_X, JxW) = MappedInterpolants(interps, x_el)

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
end

function FiniteElementContainers.stiffness(  
  physics::Mechanics, interps, u_el, x_el, state_old_q, props_el, dt
)
  (; X_q, N, ∇N_X, JxW) = MappedInterpolants(interps, x_el)

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
    DirichletBC(:displ_x, :sset_3, fixed),
    DirichletBC(:displ_y, :sset_3, fixed),
    DirichletBC(:displ_x, :sset_1, fixed),
    DirichletBC(:displ_y, :sset_1, displace),
  ]

  # pre-setup some scratch arrays
  times = TimeStepper(0., 1., 1)
  p = create_parameters(asm, physics; dirichlet_bcs=dbcs, times=times)
  # Uu = create_unknowns(asm)

  solver = NewtonSolver(IterativeLinearSolver(asm, :CgSolver))
  integrator = QuasiStaticIntegrator(solver)
  evolve!(integrator, p)
  # update_bcs!(H1Field, solver, Uu, p)

  # FiniteElementContainers.solve!(solver, Uu, p)

  pp = PostProcessor(mesh, output_file, u)
  write_times(pp, 1, 0.0)
  write_field(pp, 1, p.h1_field)
  # write_field(pp, 1, U)
  close(pp)
end

mechanics_test()
