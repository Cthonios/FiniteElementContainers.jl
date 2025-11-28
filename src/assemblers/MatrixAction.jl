"""
$(TYPEDSIGNATURES)
"""
function assemble_matrix_action!(
  assembler, func::F, Uu, Vu, p
) where F <: Function
  assemble_matrix_action!(
    assembler.stiffness_action_storage,
    assembler.dof,
    func, Uu, Vu, p
  )
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function assemble_matrix_action!(
  storage, dof, func, Uu, Vu, p
)
  fill!(storage, zero(eltype(storage)))
  fspace = function_space(dof)
  t = current_time(p.times)
  Δt = time_step(p.times)
  _update_for_assembly!(p, dof, Uu, Vu)
  for (
    conns, 
    block_physics, ref_fe,
    state_old, state_new, props
  ) in zip(
    values(fspace.elem_conns), 
    values(p.physics), values(fspace.ref_fes),
    values(p.state_old), values(p.state_new),
    values(p.properties)
  )
    backend = KA.get_backend(p.h1_field)
    _assemble_block_matrix_action!(
      backend,
      storage,
      conns,
      func,
      block_physics, ref_fe,
      p.h1_coords, t, Δt,
      p.h1_field, p.h1_field_old, p.h1_hvp_scratch_field,
      state_old, state_new, props
    )
  end
end

# CPU Implementation

"""
$(TYPEDSIGNATURES)
Assembly method for a block labelled as block_id. This is a CPU implementation
with no threading.

TODO improve typing of fields to ensure they mathc up in terms of function 
  spaces
"""
function _assemble_block_matrix_action!(
  ::KA.CPU,
  field::AbstractField, 
  conns::Connectivity,
  func::Function,
  physics::AbstractPhysics, ref_fe::ReferenceFE,
  X::AbstractField, t::T, Δt::T,
  U::Solution, U_old::Solution, V::Solution,
  state_old::S, state_new::S, props::AbstractArray
) where {
  T        <: Number,
  Solution <: AbstractField,
  S        <: L2QuadratureField
}
  for e in axes(conns, 2)
    x_el = _element_level_fields_flat(X, ref_fe, conns, e)
    u_el = _element_level_fields_flat(U, ref_fe, conns, e)
    u_el_old = _element_level_fields_flat(U_old, ref_fe, conns, e)
    v_el = _element_level_fields_flat(V, ref_fe, conns, e)
    props_el = _element_level_properties(props, e)
    K_el = _element_scratch_matrix(ref_fe, U)
    for q in 1:num_quadrature_points(ref_fe)
      interps = _cell_interpolants(ref_fe, q)
      state_old_q = _quadrature_level_state(state_old, q, e)
      K_q, state_new_q = func(physics, interps, x_el, t, Δt, u_el, u_el_old, state_old_q, props_el)
      K_el = K_el + K_q
      # update state here
      for s in 1:length(state_old)
        state_new[s, q, e] = state_new_q[s]
      end
    end
    Kv_el = K_el * v_el
    @views _assemble_element!(field, Kv_el, conns[:, e])
  end
end

# GPU implementation

"""
$(TYPEDSIGNATURES)
Kernel for residual block assembly

TODO mark const fields
"""
# COV_EXCL_START
KA.@kernel function _assemble_block_matrix_action_kernel!(
  field::AbstractField, 
  conns::Connectivity,
  func::Function,
  physics::AbstractPhysics, ref_fe::ReferenceFE,
  X::AbstractField, t::T, Δt::T,
  U::Solution, U_old::Solution, V::Solution,
  state_old::S, state_new::S, props::AbstractArray
) where {
  T        <: Number,
  Solution <: AbstractField,
  S        <: L2QuadratureField
}
  E = KA.@index(Global)

  x_el = _element_level_fields_flat(X, ref_fe, conns, E)
  u_el = _element_level_fields_flat(U, ref_fe, conns, E)
  u_el_old = _element_level_fields_flat(U_old, ref_fe, conns, E)
  v_el = _element_level_fields_flat(V, ref_fe, conns, E)
  props_el = _element_level_properties(props, E)
  K_el = _element_scratch_matrix(ref_fe, U)
  for q in 1:num_quadrature_points(ref_fe)
    interps = _cell_interpolants(ref_fe, q)
    state_old_q = _quadrature_level_state(state_old, q, E)
    K_q, state_new_q = func(physics, interps, x_el, t, Δt, u_el, u_el_old, state_old_q, props_el)
    K_el = K_el + K_q
    # update state here
    for s in 1:length(state_old)
      state_new[s, q, E] = state_new_q[s]
    end
  end
  Kv_el = K_el * v_el

  # now assemble atomically
  n_dofs = size(field, 1)
  for i in 1:size(conns, 1)
    for d in 1:n_dofs
      global_id = n_dofs * (conns[i, E] - 1) + d
      local_id = n_dofs * (i - 1) + d
      Atomix.@atomic field.vals[global_id] += Kv_el[local_id]
    end
  end
end
# COV_EXCL_STOP

"""
$(TYPEDSIGNATURES)
Assembly method for a block labelled as block_id. This is a GPU agnostic implementation
using KernelAbstractions and Atomix for eliminating race conditions

TODO add state variables and physics properties
"""
function _assemble_block_matrix_action!(
  backend::KA.Backend,
  field::AbstractField, 
  conns::Connectivity,
  func::Function,
  physics::AbstractPhysics, ref_fe::ReferenceFE,
  X::AbstractField, t::T, Δt::T,
  U::Solution, U_old::Solution, V::Solution,
  state_old::S, state_new::S, props::AbstractArray
) where {
  T        <: Number,
  Solution <: AbstractField,
  S        <: L2QuadratureField
}
  kernel! = _assemble_block_matrix_action_kernel!(backend)
  kernel!(
    field, 
    conns,
    func,
    physics, ref_fe, 
    X, t, Δt,
    U, U_old, V, 
    state_old, state_new, props, t, Δt,
    ndrange=size(conns, 2)
  )
  return nothing
end
