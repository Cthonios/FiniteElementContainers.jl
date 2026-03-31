# since this type of physics has no state
# just return the old state which will be empty

struct Laplace <: AbstractPhysics{1, 0, 0}
end

@inline function FiniteElementContainers.energy(
  physics::Laplace, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  u_q, ∇u_q = interpolate_field_values_and_gradients(physics, interps, u_el)
  e_q = 0.5 * dot(∇u_q, ∇u_q)
  return JxW * e_q
end

@inline function FiniteElementContainers.mass(
  physics::Laplace, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  M_q = N * N'
  return JxW * M_q
end

@inline function FiniteElementContainers.residual(
  physics::Laplace, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  ∇u_q = interpolate_field_gradients(physics, interps, u_el)
  R_q = ∇u_q * ∇N_X'
  return JxW * R_q[:]
end

@inline function FiniteElementContainers.stiffness(
  physics::Laplace, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  K_q = ∇N_X * ∇N_X'
  return JxW * K_q
end

@inline function FiniteElementContainers.stiffness_action(
  physics::Laplace, interps, x_el, t, dt, u_el, u_el_old, v_el, state_old_q, state_new_q, props_el
)
  interps = map_interpolants(interps, x_el)
  (; ∇N_X, JxW) = interps
  return JxW * ∇N_X * (∇N_X' * v_el)
end

@inline function FiniteElementContainers.mass_action(
  physics::Laplace, interps, x_el, t, dt, u_el, u_el_old, v_el, state_old_q, state_new_q, props_el
)
  interps = map_interpolants(interps, x_el)
  (; N, JxW) = interps
  return JxW * dot(N, v_el) * N
end
