# since this type of physics has no state
# just return the old state which will be empty

struct Poisson <: AbstractPhysics{1, 0, 0}
end

@inline function FiniteElementContainers.residual(
  physics::Poisson, interps, u_el, x_el, state_old_q, props_el, t, dt
)
  interps = MappedInterpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  ∇u_q = interpolate_field_gradients(physics, interps, u_el)
  R_q = ∇u_q * ∇N_X' - N' * f(X_q, 0.0)
  return JxW * R_q[:], state_old_q
end

@inline function FiniteElementContainers.stiffness(
  physics::Poisson, interps, u_el, x_el, state_old_q, props_el, t, dt
)
  interps = MappedInterpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  K_q = ∇N_X * ∇N_X'
  return JxW * K_q, state_old_q
end
