function assemble_scalar!(assembler, Uu, p, ::Type{H1Field}, func::F) where F <: Function
  fspace = function_space(assembler, H1Field)
  t = current_time(p.times)
  Δt = time_step(p.times)
  update_bcs!(p)
  update_field_unknowns!(p.h1_field, assembler.dof, Uu)
  for (b, (field, conns, block_physics, state_old, state_new, props)) in enumerate(zip(
    values(assembler.scalar_quadarature_storage),
    values(fspace.elem_conns), 
    values(p.physics),
    values(p.state_old), values(p.state_new),
    values(p.properties)
  ))
    ref_fe = values(fspace.ref_fes)[b]
    backend = _check_backends(assembler, p.h1_field, p.h1_coords, state_old, state_new, conns)
    _assemble_block_scalar!(
      field, block_physics, ref_fe, 
      p.h1_field, p.h1_coords, state_old, state_new, props, t, Δt,
      conns, b, func,
      backend
    )
    KA.synchronize(backend)
  end
end

function assemble!(assembler, Uu, p, ::Val{:energy}, ::Type{H1Field})
  fspace = function_space(assembler, H1Field)
  t = current_time(p.times)
  Δt = time_step(p.times)
  update_bcs!(p)
  update_field_unknowns!(p.h1_field, assembler.dof, Uu)
  for (b, (field, conns, block_physics, state_old, state_new, props)) in enumerate(zip(
    values(assembler.scalar_quadarature_storage),
    values(fspace.elem_conns), 
    values(p.physics),
    values(p.state_old), values(p.state_new),
    values(p.properties)
  ))
    ref_fe = values(fspace.ref_fes)[b]
    backend = _check_backends(assembler, p.h1_field, p.h1_coords, state_old, state_new, conns)
    _assemble_block_scalar!(
      field, block_physics, ref_fe, 
      p.h1_field, p.h1_coords, state_old, state_new, props, t, Δt,
      conns, b, energy,
      backend
    )
    KA.synchronize(backend)
  end

  # TODO need to eventually sum that all up somewhere
end

# CPU Implementation

"""
$(TYPEDSIGNATURES)
Assembly method for a block labelled as block_id. This is a CPU implementation
with no threading.

TODO improve typing of fields to ensure they mathc up in terms of function 
  spaces
"""
function _assemble_block_scalar!(
  field::F1, physics::Phys, ref_fe::R, 
  U::F2, X::F3, state_old::S, state_new::S, props::P, t::T, Δt::T,
  conns::C, block_id::Int, func::Func, ::KA.CPU
) where {
  C    <: Connectivity,
  F1   <: AbstractArray{<:Number, 2},
  F2   <: AbstractField,
  F3   <: AbstractField,
  P    <: Union{<:SVector, <:L2ElementField},
  Func <: Function,
  Phys <: AbstractPhysics, 
  R    <: ReferenceFE,
  S    <: L2QuadratureField,
  T    <: Number
}
  for e in axes(conns, 2)
    x_el = _element_level_fields_flat(X, ref_fe, conns, e)
    u_el = _element_level_fields_flat(U, ref_fe, conns, e)
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
KA.@kernel function _assemble_block_scalar_kernel!(
  field::F1, physics::Phys, ref_fe::R, 
  U::F2, X::F3, state_old::S, state_new::S, props::P, t::T, Δt::T,
  conns::C, block_id::Int, func::Func
) where {
  C    <: Connectivity,
  F1   <: AbstractArray{<:Number, 2},
  F2   <: AbstractField,
  F3   <: AbstractField,
  Func <: Function,
  P    <: Union{<:SVector, <:L2ElementField},
  Phys <: AbstractPhysics,
  R    <: ReferenceFE,
  S    <: L2QuadratureField,
  T    <: Number
}
  # Q, E = KA.@index(Global, NTuple)
  E = KA.@index(Global)
  x_el = _element_level_fields_flat(X, ref_fe, conns, E)
  u_el = _element_level_fields_flat(U, ref_fe, conns, E)
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

"""
$(TYPEDSIGNATURES)
Assembly method for a block labelled as block_id. This is a GPU agnostic implementation
using KernelAbstractions and Atomix for eliminating race conditions

TODO add state variables and physics properties
"""
function _assemble_block_scalar!(
  field::F1, physics::Phys, ref_fe::R, 
  U::F2, X::F3, state_old::S, state_new::S, props::P, t::T, Δt::T,
  conns::C, block_id::Int, func::Func, backend::KA.Backend
) where {
  C    <: Connectivity,
  F1   <: AbstractArray{<:Number, 2},
  F2   <: AbstractField,
  F3   <: AbstractField,
  Func <: Function,
  P    <: Union{<:SVector, <:L2ElementField},
  Phys <: AbstractPhysics,
  R    <: ReferenceFE,
  S    <: L2QuadratureField,
  T    <: Number
}
  kernel! = _assemble_block_scalar_kernel!(backend)
  kernel!(
    field, physics, ref_fe, 
    U, X, state_old, state_new, props, t, Δt,
    conns, block_id, func, ndrange=size(field, 2)
  )
  # KA.synchronize(backend)
  return nothing
end
