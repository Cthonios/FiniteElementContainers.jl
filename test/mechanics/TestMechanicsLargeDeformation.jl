using Exodus
using FiniteElementContainers
using Tensors

# mesh file
gold_file = "./test/mechanics/mechanics.gold"
mesh_file = "./test/mechanics/mechanics.g"
output_file = "./test/mechanics/mechanics.e"

fixed(_, _) = 0.
displace(_, t) = 0.1 * t

struct Mechanics{Form} <: AbstractPhysics{2, 0, 0}
  formulation::Form
end

function strain_energy(∇u)
  K = 10.e6
  G = 1.e6
  I = one(Tensor{2, 3, eltype(∇u), 9})
  F = I + ∇u
  B = dott(F)
  J = det(F)
  I1_bar = J^(-2. / 3.) * tr(B)
  ψ = 0.5 * K * (0.5 * (J - 1)^2 - log(J)) + 
      0.5 * G * (I1_bar - 3.)
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
  times = TimeStepper(0., 1., 100)
  p = create_parameters(asm, physics; dirichlet_bcs=dbcs, times=times)
  # Uu = create_unknowns(asm)

  solver = NewtonSolver(IterativeLinearSolver(asm, :CgSolver))
  integrator = QuasiStaticIntegrator(solver)
  pp = PostProcessor(mesh, output_file, u)

  for n in 1:100
    @info "Time step $n"
    evolve!(integrator, p)
    write_times(pp, n, p.times.time_current[1])
    write_field(pp, n, p.h1_field)
  end
  # update_bcs!(H1Field, solver, Uu, p)

  # FiniteElementContainers.solve!(solver, Uu, p)

  
  # write_field(pp, 1, U)
  close(pp)
# end

# mechanics_test()
