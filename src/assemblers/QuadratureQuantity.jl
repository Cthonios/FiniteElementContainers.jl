"""
$(TYPEDSIGNATURES)
"""
function assemble_scalar!(
  assembler, func::F, Uu, p
) where F <: Function
  assemble_quadrature_quantity!(
    assembler.scalar_quadrature_storage, assembler.dof,
    func, Uu, p
  )
end

"""
$(TYPEDSIGNATURES)
"""
function assemble_quadrature_quantity!(
  storage, dof,
  func::F, Uu, p
) where F <: Function
  fspace = function_space(dof)
  t = current_time(p.times)
  Δt = time_step(p.times)
  _update_for_assembly!(p, dof, Uu)
  for (
    block_storage,
    conns, 
    block_physics, ref_fe,
    state_old, state_new, props
  ) in zip(
    values(storage),
    values(fspace.elem_conns), 
    values(p.physics), values(fspace.ref_fes),
    values(p.state_old), values(p.state_new),
    values(p.properties)
  )
    backend = KA.get_backend(p.h1_field)
    _assemble_block_quadrature_quantity!(
      backend,
      block_storage,
      conns,
      func,
      block_physics, ref_fe,
      p.h1_coords, t, Δt,
      p.h1_field, p.h1_field_old,
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
function _assemble_block_quadrature_quantity!(
  ::KA.CPU,
  field::AbstractArray, 
  conns::Connectivity,
  func::Function,
  physics::AbstractPhysics, ref_fe::ReferenceFE,
  X::AbstractField, t::T, Δt::T,
  U::Solution, U_old::Solution,
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
    props_el = _element_level_properties(props, e)

    for q in 1:num_quadrature_points(ref_fe)
      interps = _cell_interpolants(ref_fe, q)
      state_old_q = _quadrature_level_state(state_old, q, e)
      e_q, state_new_q = func(physics, interps, x_el, t, Δt, u_el, u_el_old, state_old_q, props_el)
      field[q, e] = e_q
      # update state here
      for s in 1:length(state_old)
        state_new[s, q, e] = state_new_q[s]
      end
    end
  end
end

"""
$(TYPEDSIGNATURES)
Kernel for residual block assembly

TODO mark const fields
"""
# COV_EXCL_START
KA.@kernel function _assemble_block_quadrature_quantity_kernel!(
  field::AbstractArray, 
  conns::Connectivity,
  func::Function,
  physics::AbstractPhysics, ref_fe::ReferenceFE,
  X::AbstractField, t::T, Δt::T,
  U::Solution, U_old::Solution,
  state_old::S, state_new::S, props::AbstractArray
) where {
  T        <: Number,
  Solution <: AbstractField,
  S        <: L2QuadratureField
}
  # Q, E = KA.@index(Global, NTuple)
  E = KA.@index(Global)
  x_el = _element_level_fields_flat(X, ref_fe, conns, E)
  u_el = _element_level_fields_flat(U, ref_fe, conns, E)
  u_el_old = _element_level_fields_flat(U_old, ref_fe, conns, E)
  props_el = _element_level_properties(props, E)

  KA.Extras.@unroll for q in 1:num_quadrature_points(ref_fe)
    interps = _cell_interpolants(ref_fe, q)
    state_old_q = _quadrature_level_state(state_old, q, E)
    e_q, state_new_q = func(physics, interps, x_el, t, Δt, u_el, u_el_old, state_old_q, props_el)
    @inbounds field[q, E] = e_q
    for s in 1:length(state_old)
      @inbounds state_new[s, q, E] = state_new_q[s]
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
function _assemble_block_quadrature_quantity!(
  backend::KA.Backend,
  field::AbstractArray, 
  conns::Connectivity,
  func::Function,
  physics::AbstractPhysics, ref_fe::ReferenceFE,
  X::AbstractField, t::T, Δt::T,
  U::Solution, U_old::Solution,
  state_old::S, state_new::S, props::AbstractArray
) where {
  T        <: Number,
  Solution <: AbstractField,
  S        <: L2QuadratureField
}
  kernel! = _assemble_block_quadrature_quantity_kernel!(backend)
  kernel!(
    field, 
    conns,
    func,
    physics, ref_fe, 
    X, t, Δt,
    U, U_old, 
    state_old, state_new, props,
    ndrange=size(field, 2)
  )
  # KA.synchronize(backend)
  return nothing
end
