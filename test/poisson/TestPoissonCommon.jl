# since this type of physics has no state
# just return the old state which will be empty

struct Poisson{F <: Function} <: AbstractPhysics{1, 0, 0}
  func::F
end

@inline function FiniteElementContainers.energy(
  physics::Poisson, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  u_q, ∇u_q = interpolate_field_values_and_gradients(physics, interps, u_el)
  e_q = 0.5 * dot(∇u_q, ∇u_q) - dot(u_q, physics.func(X_q, 0.0))
  return JxW * e_q
end

@inline function FiniteElementContainers.mass(
  physics::Poisson, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  M_q = N * N'
  return JxW * M_q
end

@inline function FiniteElementContainers.residual(
  physics::Poisson, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
  # interps = MappedInterpolants(interps, x_el)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  ∇u_q = interpolate_field_gradients(physics, interps, u_el)
  R_q = ∇u_q * ∇N_X' - N' * physics.func(X_q, 0.0)
  return JxW * R_q[:]
end

@inline function FiniteElementContainers.stiffness(
  physics::Poisson, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
  # interps = MappedInterpolants(interps, x_el)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  K_q = ∇N_X * ∇N_X'
  return JxW * K_q
end
