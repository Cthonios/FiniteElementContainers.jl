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
  cell = map_interpolants(interps, x_el)
  (; N, JxW) = cell

  N_nodes = size(N, 1)
  NDIM = 1
  NDOF = NDIM * N_nodes
  tup = zeros(SVector{NDOF * NDOF, eltype(N)})
  for n in 1:N_nodes
    for m in 1:N_nodes
      Nnm = N[n] * N[m]
      for d in 1:NDIM
          r = NDIM * (n - 1) + d
          c = NDIM * (m - 1) + d
          linear_idx = r + NDOF * (c - 1)   # column-major flat index
          tup = setindex(tup, Nnm, linear_idx)
      end
    end
  end
  M_el = SMatrix{NDOF, NDOF, eltype(N), NDOF * NDOF}(tup.data)
  return JxW * M_el
end

@inline function FiniteElementContainers.mass!(
  storage, e,
  physics::Poisson, t, dt, props_el, 
  state_old_q, state_new_q,
  conn, interps, x_el, u_el, u_el_old
)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  form = GeneralFormulation{size(X_q, 1), num_fields(physics)}()
  scatter_with_values_and_values!(storage, form, e, conn, N, JxW)
end

@inline function FiniteElementContainers.mass_action!(
  storage, e,
  physics::Poisson, t, dt, props_el, 
  state_old_q, state_new_q,
  conn, interps, x_el, u_el, u_el_old, v_el
)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  form = GeneralFormulation{size(X_q, 1), num_fields(physics)}()
  scatter_with_values_and_values!(storage, form, e, conn, N, JxW, v_el)
end

@inline function FiniteElementContainers.mass_action(
  physics::Poisson, interps, x_el, t, dt, u_el, u_el_old, v_el, state_old_q, state_new_q, props_el
)
  interps = map_interpolants(interps, x_el)
  (; N, JxW) = interps
  return JxW * dot(N, v_el) * N
end

@inline function FiniteElementContainers.residual(
  physics::Poisson, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  ∇u_q = interpolate_field_gradients(physics, interps, u_el)
  R_q = ∇u_q * ∇N_X' - N' * physics.func(X_q, 0.0)
  return JxW * R_q[:]
end

@inline function FiniteElementContainers.residual!(
  storage, e,
  physics::Poisson, t, dt, props_el, 
  state_old_q, state_new_q,
  conn, interps, x_el, u_el, u_el_old
)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  ∇u_q = interpolate_field_gradients(physics, interps, u_el)
  # R_q = ∇u_q * ∇N_X' - N' * physics.func(X_q, 0.0)
  form = GeneralFormulation{size(X_q, 1), num_fields(physics)}()
  scatter_with_gradients!(storage, form, e, conn, ∇N_X, JxW * ∇u_q)
  scatter_with_values!(storage, form, e, conn, N, -JxW * physics.func(X_q, 0.0))
end

@inline function FiniteElementContainers.stiffness(
  physics::Poisson, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el
)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  K_q = ∇N_X * ∇N_X'
  return JxW * K_q
end

@inline function FiniteElementContainers.stiffness!(
  storage, e,
  physics::Poisson, t, dt, props_el, 
  state_old_q, state_new_q,
  conn, interps, x_el, u_el, u_el_old
)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  form = GeneralFormulation{size(X_q, 1), num_fields(physics)}()
  scatter_with_gradients_and_gradients!(storage, form, e, conn, ∇N_X, JxW)
end

@inline function FiniteElementContainers.stiffness_action(
  physics::Poisson, interps, x_el, t, dt, u_el, u_el_old, v_el, state_old_q, state_new_q, props_el
)
  interps = map_interpolants(interps, x_el)
  (; ∇N_X, JxW) = interps
  return JxW * ∇N_X * (∇N_X' * v_el)
end

@inline function FiniteElementContainers.stiffness_action!(
  storage, e,
  physics::Poisson, t, dt, props_el, 
  state_old_q, state_new_q,
  conn, interps, x_el, u_el, u_el_old, v_el
)
  interps = map_interpolants(interps, x_el)
  (; X_q, N, ∇N_X, JxW) = interps
  form = GeneralFormulation{size(X_q, 1), num_fields(physics)}()
  scatter_with_gradients_and_gradients!(storage, form, e, conn, ∇N_X, JxW, v_el)
end
