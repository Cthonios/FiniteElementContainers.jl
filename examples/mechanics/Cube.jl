using Exodus
using FiniteElementContainers
using Tensors

fixed(_, _) = 0.
displace(_, t) = 1.0 * t

struct Mechanics{Form} <: AbstractPhysics{2, 0, 0}
  formulation::Form
end

function strain_energy(∇u)
  E = 1.e9
  ν = 0.25
  K = E / (3. * (1. - 2. * ν))
  G = E / (2. * (1. + ν))

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
  mesh = UnstructuredMesh("cube.g")
  V = FunctionSpace(mesh, H1Field, Lagrange) 
  physics = (;
    cube=Mechanics(ThreeDimensional())
  )

  u = VectorFunction(V, :displ)
  asm = SparseMatrixAssembler(H1Field, u)

  dbcs = DirichletBC[
    DirichletBC("displ_x", "ssx-", fixed),
    DirichletBC("displ_y", "ssy-", fixed),
    DirichletBC("displ_z", "ssz-", fixed),
    DirichletBC("displ_z", "ssz+", displace),
  ]

  # pre-setup some scratch arrays
  times = TimeStepper(0., 1., 10)
  p = create_parameters(asm, physics; dirichlet_bcs=dbcs, times=times)

  solver = NewtonSolver(IterativeLinearSolver(asm, :CgSolver))
  integrator = QuasiStaticIntegrator(solver)
  pp = PostProcessor(mesh, "cube.e", u)

  write_times(pp, 1, p.times.time_current[1])
  write_field(pp, 1, p.h1_field)

  for n in 1:10
    evolve!(integrator, p)
    write_times(pp, n, p.times.time_current[1])
    write_field(pp, n, p.h1_field)
  end
  close(pp)
end

mechanics_test()
