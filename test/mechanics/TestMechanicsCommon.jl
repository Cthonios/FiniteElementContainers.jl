# using Enzyme

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

@inline function pk1_stress(
  ∇u, state_old, props, dt
)
  K, G = props[1], props[2]
  ε = symmetric(∇u)
  # ε_dev = dev(ε)
  I = one(Tensor{2, 3, Float64, 9})
  F = ∇u + I
  J = det(F)
  # @show I
  # @show dev(ε)
  σ = K * tr(ε) * I #+ 2. * G + dev(ε)
  P = J * dot(σ, inv(F)')
  return P
end

@inline function FiniteElementContainers.energy(
  physics::Mechanics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  ∇u_q = interpolate_field_gradients(physics, interps, u_el)

  # kinematics
  ∇u_q = modify_field_gradients(physics.formulation, ∇u_q)
  # constitutive
  ψ_q = strain_energy(∇u_q, state_old_q, props_el, dt)
  return JxW * ψ_q
end

# note for CUDA things crash without inline
@inline function FiniteElementContainers.residual(
  physics::Mechanics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  ∇u_q = interpolate_field_gradients(physics, interps, u_el)

  # kinematics
  ∇u_q = modify_field_gradients(physics.formulation, ∇u_q)
  # constitutive
  P_q = gradient(z -> strain_energy(z, state_old_q, props_el, dt), ∇u_q)
  # turn into voigt notation
  P_q = extract_stress(physics.formulation, P_q)
  G_q = discrete_gradient(physics.formulation, ∇N_X)
  f_q = G_q * P_q
  return JxW * f_q[:]
end

@inline function FiniteElementContainers.residual!(
  storage, e,
  physics::Mechanics, t, dt, props_el, 
  state_old_q, state_new_q,
  conn, interps, x_el, u_el, u_el_old
)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  ∇u_q = interpolate_field_gradients(physics, interps, u_el)

  # kinematics
  ∇u_q = modify_field_gradients(physics.formulation, ∇u_q)
  # constitutive
  P_q = gradient(z -> strain_energy(z, state_old_q, props_el, dt), ∇u_q)
  scatter_with_gradients!(storage, physics.formulation, e, conn, ∇N_X, JxW * P_q)
end

@inline function FiniteElementContainers.stiffness(
  physics::Mechanics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  ∇u_q = interpolate_field_gradients(physics, interps, u_el)

  # kinematics
  ∇u_q = modify_field_gradients(physics.formulation, ∇u_q)
  # constitutive
  K_q = hessian(z -> strain_energy(z, state_old_q, props_el, dt), ∇u_q)
  # turn into voigt notation
  K_q = extract_stiffness(physics.formulation, K_q)
  G_q = discrete_gradient(physics.formulation, ∇N_X)
  return JxW * G_q * K_q * G_q'
end

@inline function FiniteElementContainers.stiffness!(
  storage, e,
  physics::Mechanics, t, dt, props_el, 
  state_old_q, state_new_q,
  conn, interps, x_el, u_el, u_el_old
)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  ∇u_q = interpolate_field_gradients(physics, interps, u_el)

  # kinematics
  ∇u_q = modify_field_gradients(physics.formulation, ∇u_q)
  # constitutive
  K_q = hessian(z -> strain_energy(z, state_old_q, props_el, dt), ∇u_q)
  scatter_with_gradients_and_gradients!(
    storage, physics.formulation, e, conn, ∇N_X, JxW * K_q
  )
end

@inline function FiniteElementContainers.stiffness_action(
  physics::Mechanics, interps, x_el, t, dt, u_el, u_el_old, v_el, state_old_q, state_new_q, props_el
)
  interps = map_interpolants(interps, x_el)
  (; ∇N_X, JxW) = interps
  ∇u_q = interpolate_field_gradients(physics, interps, u_el)
  ∇u_q = modify_field_gradients(physics.formulation, ∇u_q)
  K_q = hessian(z -> strain_energy(z, state_old_q, props_el, dt), ∇u_q)
  K_q = extract_stiffness(physics.formulation, K_q)
  G_q = discrete_gradient(physics.formulation, ∇N_X)
  return JxW * G_q * (K_q * (G_q' * v_el))
end
