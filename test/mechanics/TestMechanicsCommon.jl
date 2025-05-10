struct Mechanics{Form} <: AbstractPhysics{2, 2, 0}
  formulation::Form
end

function FiniteElementContainers.create_properties(::Mechanics)
  K = 10.e9
  G = 1.e9
  return SVector{2, Float64}(K, G)
end

@inline function strain_energy(
  ∇u, state_old, props, dt
)
  K, G = props[1], props[2]
  ε = symmetric(∇u)
  ψ = 0.5 * K * tr(ε)^2 + G * dcontract(dev(ε), dev(ε))
end

# note for CUDA things crash without inline
@inline function FiniteElementContainers.residual(
  physics::Mechanics, interps, u_el, x_el, state_old_q, props_el, dt
)
  (; X_q, N, ∇N_X, JxW) = MappedInterpolants(interps, x_el)

  # kinematics
  ∇u_q = u_el * ∇N_X
  ∇u_q = modify_field_gradients(physics.formulation, ∇u_q)
  # constitutive
  P_q = gradient(z -> strain_energy(z, state_old_q, props_el, dt), ∇u_q)
  # turn into voigt notation
  P_q = extract_stress(physics.formulation, P_q)
  G_q = discrete_gradient(physics.formulation, ∇N_X)
  f_q = G_q * P_q
  return JxW * f_q[:], state_old_q
end

@inline function FiniteElementContainers.stiffness(  
  physics::Mechanics, interps, u_el, x_el, state_old_q, props_el, dt
)
  (; X_q, N, ∇N_X, JxW) = MappedInterpolants(interps, x_el)

  # kinematics
  ∇u_q = u_el * ∇N_X
  ∇u_q = modify_field_gradients(physics.formulation, ∇u_q)
  # constitutive
  K_q = hessian(z -> strain_energy(z, state_old_q, props_el, dt), ∇u_q)
  # turn into voigt notation
  K_q = extract_stiffness(physics.formulation, K_q)
  G_q = discrete_gradient(physics.formulation, ∇N_X)
  return JxW * G_q * K_q * G_q', state_old_q
end
