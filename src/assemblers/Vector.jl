"""
$(TYPEDSIGNATURES)
"""
function assemble_vector!(
  assembler, func::F, Uu, p, ::Type{H1Field}
) where F <: Function
  storage = assembler.residual_storage
  fill!(storage, zero(eltype(storage)))
  fspace = function_space(assembler, H1Field)
  t = current_time(p.times)
  Δt = time_step(p.times)
  update_bcs!(p)
  update_field_unknowns!(p.h1_field, assembler.dof, Uu)
  for (b, (conns, block_physics, state_old, state_new, props)) in enumerate(zip(
    values(fspace.elem_conns), 
    values(p.physics),
    values(p.state_old), values(p.state_new),
    values(p.properties)
  ))
    ref_fe = values(fspace.ref_fes)[b]
    backend = _check_backends(assembler, p.h1_field, p.h1_coords, state_old, state_new, conns)
    _assemble_block_vector!(
      storage, block_physics, ref_fe, 
      p.h1_field, p.h1_field_old, p.h1_coords, state_old, state_new, props, t, Δt,
      conns, b, 
      func,
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
function _assemble_block_vector!(
  field::F1, physics::Phys, ref_fe::R, 
  U::F2, U_old::F2, X::F3, state_old::S, state_new::S, props::P, t::T, Δt::T,
  conns::C, block_id::Int, 
  func::Func, ::KA.CPU
) where {
  C    <: Connectivity,
  F1   <: AbstractField,
  F2   <: AbstractField,
  F3   <: AbstractField,
  P    <: Union{<:SVector, <:L2ElementField},
  Func <: Function,
  Phys <: AbstractPhysics, 
  R    <: ReferenceFE,
  S    <: L2QuadratureField,
  T    <: Number
}
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
  for e in axes(conns, 2)
    x_el = _element_level_fields_flat(X, ref_fe, conns, e)
    u_el = _element_level_fields_flat(U, ref_fe, conns, e)
    u_el_old = _element_level_fields_flat(U_old, ref_fe, conns, e)
    props_el = _element_level_properties(props, e)
    R_el = zeros(SVector{NxNDof, eltype(field)})

    for q in 1:num_quadrature_points(ref_fe)
      interps = _cell_interpolants(ref_fe, q)
      state_old_q = _quadrature_level_state(state_old, q, e)
      R_q, state_new_q = func(physics, interps, u_el, x_el, state_old_q, props_el, t, Δt)
      R_el = R_el + R_q
      # update state here
      for s in 1:length(state_old)
        state_new[s, q, e] = state_new_q[s]
      end
    end
    
    @views _assemble_element!(field, R_el, conns[:, e], e, block_id)
  end
end

# TODO hardcoded to H1 fields right now.
"""
$(TYPEDSIGNATURES)
"""
function _residual(asm::AbstractAssembler, ::KA.CPU)\
  @views asm.residual_unknowns .= asm.residual_storage[asm.dof.H1_unknown_dofs]
  return asm.residual_unknowns
end

# GPU implementation

"""
$(TYPEDSIGNATURES)
Kernel for residual block assembly

TODO mark const fields
"""
# COV_EXCL_START
KA.@kernel function _assemble_block_vector_kernel!(
  field::F1, physics::Phys, ref_fe::R, 
  U::F2, U_old::F2, X::F3, state_old::S, state_new::S, props::P, t::T, Δt::T,
  conns::C, block_id::Int, 
  func::Func
) where {
  C    <: Connectivity,
  F1   <: AbstractField,
  F2   <: AbstractField,
  F3   <: AbstractField,
  Func <: Function,
  P    <: Union{<:SVector, <:L2ElementField},
  Phys <: AbstractPhysics,
  R    <: ReferenceFE,
  S    <: L2QuadratureField,
  T    <: Number
}
  E = KA.@index(Global)

  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND

  x_el = _element_level_fields_flat(X, ref_fe, conns, E)
  u_el = _element_level_fields_flat(U, ref_fe, conns, E)
  u_el_old = _element_level_fields_flat(U_old, ref_fe, conns, E)
  props_el = _element_level_properties(props, E)
  R_el = zeros(SVector{NxNDof, eltype(field)})

  for q in 1:num_quadrature_points(ref_fe)
    interps = _cell_interpolants(ref_fe, q)
    state_old_q = _quadrature_level_state(state_old, q, E)
    R_q, state_new_q = func(physics, interps, u_el, x_el, state_old_q, props_el, t, Δt)
    R_el = R_el + R_q
    # update state here
    for s in 1:length(state_old)
      state_new[s, q, E] = state_new_q[s]
    end
  end

  # now assemble atomically
  n_dofs = size(field, 1)
  for i in 1:size(conns, 1)
    for d in 1:n_dofs
      global_id = n_dofs * (conns[i, E] - 1) + d
      local_id = n_dofs * (i - 1) + d
      Atomix.@atomic field.vals[global_id] += R_el[local_id]
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
function _assemble_block_vector!(
  field::F1, physics::Phys, ref_fe::R, 
  U::F2, U_old::F2, X::F3, state_old::S, state_new::S, props::P, t::T, Δt::T,
  conns::C, block_id::Int, 
  func::Func, backend::KA.Backend
) where {
  C    <: Connectivity,
  F1   <: AbstractField,
  F2   <: AbstractField,
  F3   <: AbstractField,
  Func <: Function,
  P    <: Union{<:SVector, <:L2ElementField},
  Phys <: AbstractPhysics,
  R    <: ReferenceFE,
  S    <: L2QuadratureField,
  T    <: Number
}
  kernel! = _assemble_block_vector_kernel!(backend)
  kernel!(
    field, physics, ref_fe, 
    U, U_old, X, state_old, state_new, props, t, Δt,
    conns, block_id,
    func, ndrange=size(conns, 2)
  )
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
# COV_EXCL_START
KA.@kernel function _extract_residual_unknowns!(Ru, unknown_dofs, R)
  N = KA.@index(Global)
  Ru[N] = R[unknown_dofs[N]]
end
# COV_EXCL_STOP

"""
$(TYPEDSIGNATURES)
"""
function _residual(asm::AbstractAssembler, backend::KA.Backend)
  kernel! = _extract_residual_unknowns!(backend)
  kernel!(
    asm.residual_unknowns, 
    asm.dof.H1_unknown_dofs, 
    asm.residual_storage, 
    ndrange=length(asm.dof.H1_unknown_dofs)
  )
  KA.synchronize(KA.get_backend(asm))
  return asm.residual_unknowns
end 
