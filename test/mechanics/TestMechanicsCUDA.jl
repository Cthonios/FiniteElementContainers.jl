using Adapt
using CUDA
using Exodus
using FiniteElementContainers
using StaticArrays
using Tensors

# mesh file
gold_file = "./test/mechanics/mechanics.gold"
mesh_file = "./test/mechanics/mechanics.g"
output_file = "./test/mechanics/mechanics.e"

fixed(_, _) = 0.
displace(_, t) = 1.e-3 * t

include("TestMechanicsCommon.jl")

# struct Mechanics{Form} <: AbstractPhysics{2, 2, 0}
#   formulation::Form
# end

# function FiniteElementContainers.create_properties(::Mechanics)
#   K = 10.e9
#   G = 1.e9
#   return SVector{2, Float64}(K, G)
# end

# @inline function strain_energy(
#   ∇u::Tensor{2, 3, T1, 9}, state_old::SVector{NS, T2}, props_el::SVector{NP, T3}, dt::T4
# ) where {NS, NP, T1, T2, T3, T4}
#   K, G = props[1], props[2]
#   ε = symmetric(∇u)
#   ψ = 0.5 * K * tr(ε)^2 + G * dcontract(dev(ε), dev(ε))
# end

# # @inline function pk1_stress(∇u, state_old, props, dt)
# #   return gradient(z -> strain_energy(z, state_old, props, dt), ∇u)
# # end

# # note for CUDA things crash without inline
# @inline function FiniteElementContainers.residual(
#   physics::Mechanics, interps, u_el, x_el, state_old_q, props_el, dt
# )
#   (; X_q, N, ∇N_X, JxW) = MappedInterpolants(interps, x_el)

#   # kinematics
#   ∇u_q = u_el * ∇N_X
#   ∇u_q = modify_field_gradients(physics.formulation, ∇u_q)
#   # constitutive
#   P_q = gradient(z -> strain_energy(z, state_old_q, props_el, dt), ∇u_q)
#   # P_q = pk1_stress(∇u_q, state_old_q, props_el, dt)
#   # P_q = gradient(z strain_energy, ∇u_q)
#   # turn into voigt notation
#   P_q = extract_stress(physics.formulation, P_q)
#   G_q = discrete_gradient(physics.formulation, ∇N_X)
#   f_q = G_q * P_q
#   return JxW * f_q[:]
# end

# @inline function FiniteElementContainers.stiffness(  
#   physics::Mechanics, interps, u_el, x_el, state_old_q, props_el, dt
# )
#   (; X_q, N, ∇N_X, JxW) = MappedInterpolants(interps, x_el)

#   # kinematics
#   ∇u_q = u_el * ∇N_X
#   ∇u_q = modify_field_gradients(physics.formulation, ∇u_q)
#   # constitutive
#   K_q = hessian(z -> strain_energy(z, state_old_q, props_el, dt), ∇u_q)
#   # turn into voigt notation
#   K_q = extract_stiffness(physics.formulation, K_q)
#   G_q = discrete_gradient(physics.formulation, ∇N_X)
#   return JxW * G_q * K_q * G_q'
# end

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
  times = TimeStepper(0., 1., 1)
  p = create_parameters(asm, physics; dirichlet_bcs=dbcs, times=times)

  # move to device
  p_gpu = p |> cuda
  asm_gpu = asm |> cuda

  solver = NewtonSolver(IterativeLinearSolver(asm_gpu, :CgSolver))
  integrator = QuasiStaticIntegrator(solver)
  evolve!(integrator, p_gpu)
  # Uu = create_unknowns(asm_gpu)

  # update_bcs!(H1Field, solver, Uu, p_gpu)

  # @time FiniteElementContainers.solve!(solver, Uu, p_gpu)
  # @time FiniteElementContainers.solve!(solver, Uu, p_gpu)

  p = p_gpu |> cpu

  pp = PostProcessor(mesh, output_file, u)
  write_times(pp, 1, 0.0)
  write_field(pp, 1, p.h1_field)
  close(pp)
# end

# mechanics_test()
