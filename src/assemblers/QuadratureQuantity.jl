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
  for (b, (field, conns, block_physics, state_old, state_new, props)) in enumerate(zip(
    values(storage),
    values(fspace.elem_conns), 
    values(p.physics),
    values(p.state_old), values(p.state_new),
    values(p.properties)
  ))
    ref_fe = values(fspace.ref_fes)[b]
    # backend = _check_backends(assembler, p.h1_field, p.h1_field_old, p.h1_coords, state_old, state_new, conns)
    backend = KA.get_backend(p.h1_field)
    _assemble_block_quadrature_quantity!(
      field, block_physics, ref_fe, 
      p.h1_field, p.h1_field_old, p.h1_coords, state_old, state_new, props, t, Δt,
      conns, b, func,
      backend
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
  field::F1, physics::Phys, ref_fe::R, 
  U::F2, U_old::F2, X::F3, state_old::S, state_new::S, props::P, t::T, Δt::T,
  conns::C, block_id::Int, func::Func, ::KA.CPU
) where {
  C    <: Connectivity,
  # F1   <: AbstractArray{<:Number, 2},
  F1   <: AbstractMatrix,
  F2   <: AbstractField,
  F3   <: AbstractField,
  # P    <: Union{<:SVector, <:L2ElementField},
  P    <: AbstractArray,
  Func <: Function,
  Phys <: AbstractPhysics, 
  R    <: ReferenceFE,
  S    <: L2QuadratureField,
  T    <: Number
}
  for e in axes(conns, 2)
    x_el = _element_level_fields_flat(X, ref_fe, conns, e)
    u_el = _element_level_fields_flat(U, ref_fe, conns, e)
    u_el_old = _element_level_fields_flat(U_old, ref_fe, conns, e)
    props_el = _element_level_properties(props, e)

    for q in 1:num_quadrature_points(ref_fe)
      interps = _cell_interpolants(ref_fe, q)
      state_old_q = _quadrature_level_state(state_old, q, e)
      e_q, state_new_q = func(physics, interps, u_el, x_el, state_old_q, props_el, t, Δt)
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
  field::F1, physics::Phys, ref_fe::R, 
  U::F2, U_old::F2, X::F3, state_old::S, state_new::S, props::P, t::T, Δt::T,
  conns::C, block_id::Int, func::Func
) where {
  C    <: Connectivity,
  # F1   <: AbstractArray{<:Number, 2},
  F1   <: AbstractMatrix,
  F2   <: AbstractField,
  F3   <: AbstractField,
  Func <: Function,
  # P    <: Union{<:SVector, <:L2ElementField},
  P    <: AbstractArray,
  Phys <: AbstractPhysics,
  R    <: ReferenceFE,
  S    <: L2QuadratureField,
  T    <: Number
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
    e_q, state_new_q = func(physics, interps, u_el, x_el, state_old_q, props_el, t, Δt)
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
  field::F1, physics::Phys, ref_fe::R, 
  U::F2, U_old::F2, X::F3, state_old::S, state_new::S, props::P, t::T, Δt::T,
  conns::C, block_id::Int, func::Func, backend::KA.Backend
) where {
  C    <: Connectivity,
  # F1   <: AbstractArray{<:Number, 2},
  F1   <: AbstractMatrix,
  F2   <: AbstractField,
  F3   <: AbstractField,
  Func <: Function,
  # P    <: Union{<:SVector, <:L2ElementField},
  P    <: AbstractArray,
  Phys <: AbstractPhysics,
  R    <: ReferenceFE,
  S    <: L2QuadratureField,
  T    <: Number
}
  kernel! = _assemble_block_quadrature_quantity_kernel!(backend)
  kernel!(
    field, physics, ref_fe, 
    U, U_old, X, state_old, state_new, props, t, Δt,
    conns, block_id, func, ndrange=size(field, 2)
  )
  # KA.synchronize(backend)
  return nothing
end
