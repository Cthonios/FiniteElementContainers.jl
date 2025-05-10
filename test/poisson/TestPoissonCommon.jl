# since this type of physics has no state
# just return the old state which will be empty

struct Poisson <: AbstractPhysics{1, 0, 0}
end

@inline function FiniteElementContainers.residual(
  ::Poisson, interps, u_el, x_el, state_old_q, props_el, dt
)
  (; X_q, N, ∇N_X, JxW) = MappedInterpolants(interps, x_el)
  ∇u_q = u_el * ∇N_X
  R_q = ∇u_q * ∇N_X' - N' * f(X_q, 0.0)
  return JxW * R_q[:], state_old_q
end

@inline function FiniteElementContainers.stiffness(
  ::Poisson, interps, u_el, x_el, state_old_q, props_el, dt
)
  (; X_q, N, ∇N_X, JxW) = MappedInterpolants(interps, x_el)
  K_q = ∇N_X * ∇N_X'
  return JxW * K_q, state_old_q
end
