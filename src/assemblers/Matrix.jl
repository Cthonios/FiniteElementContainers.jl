# Top level methods
function assemble!(assembler, ::Type{H1Field}, Uu, p, val_sym::Val{:mass})
  _zero_storage(assembler, val_sym)
  fspace = assembler.dof.H1_vars[1].fspace
  t = current_time(p.times)
  dt = time_step(p.times)
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
    _assemble_block_matrix!(
      assembler.mass_storage, assembler.pattern, block_physics, ref_fe,
      p.h1_field, p.h1_coords, state_old, state_new, props, t, dt,
      conns, b, mass,
      backend
    )
  end
end

function assemble!(assembler, ::Type{H1Field}, Uu, p, val_sym::Val{:stiffness})
  _zero_storage(assembler, val_sym)
  fspace = assembler.dof.H1_vars[1].fspace
  t = current_time(p.times)
  dt = time_step(p.times)
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
    _assemble_block_matrix!(
      assembler.stiffness_storage, assembler.pattern, block_physics, ref_fe,
      p.h1_field, p.h1_coords, state_old, state_new, props, t, dt,
      conns, b, stiffness,
      backend
    )
  end
end

# CPU implementation

"""
$(TYPEDSIGNATURES)
Assembly method for a block labelled as block_id. This is a CPU implementation
with no threading.

TODO add state variables and physics properties
"""
function _assemble_block_matrix!(
  field::F1, pattern::Patt, physics::Phys, ref_fe::R,
  U::F2, X::F3, state_old::S, state_new::S, props::P, t::T, dt::T,
  conns::C, block_id::Int, func::Func, ::KA.CPU
) where {
  C    <: Connectivity,
  F1   <: AbstractArray{<:Number, 1},
  F2   <: AbstractField,
  F3   <: AbstractField,
  Func <: Function,
  P    <: Union{<:SVector, <:L2ElementField},
  Patt <: SparsityPattern,
  Phys <: AbstractPhysics,
  R    <: ReferenceFE,
  S    <: L2QuadratureField,
  T    <: Number
}
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
  for e in axes(conns, 2)
    x_el = _element_level_fields(X, ref_fe, conns, e)
    u_el = _element_level_fields_flat(U, ref_fe, conns, e)
    props_el = _element_level_properties(props, e)
    K_el = zeros(SMatrix{NxNDof, NxNDof, eltype(field), NxNDof * NxNDof})

    for q in 1:num_quadrature_points(ref_fe)
      interps = ref_fe.cell_interps.vals[q]
      state_old_q = _quadrature_level_state(state_old, q, e)
      K_q, state_new_q = func(physics, interps, u_el, x_el, state_old_q, props_el, t, dt)
      K_el = K_el + K_q
      for s in 1:length(state_old)
        state_new[s, q, e] = state_new_q[s]
      end
    end
    
    @views _assemble_element!(pattern, field, K_el, e, block_id)
  end
  return nothing
end


# GPU implementation

KA.@kernel function _assemble_block_matrix_kernel!(
  field::F1, pattern::Patt, physics::Phys, ref_fe::R,
  U::F2, X::F3, state_old::S, state_new::S, props::P, t::T, dt::T,
  conns::C, block_id::Int, func::Func
) where {
  C    <: Connectivity,
  F1   <: AbstractArray{<:Number, 1},
  F2   <: AbstractField,
  F3   <: AbstractField,
  Func <: Function,
  P    <: Union{<:SVector, <:L2ElementField},
  Patt <: SparsityPattern,
  Phys <: AbstractPhysics,
  R    <: ReferenceFE,
  S    <: L2QuadratureField,
  T    <: Number
}
  E = KA.@index(Global)

  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND

  x_el = _element_level_fields(X, ref_fe, conns, E)
  u_el = _element_level_fields_flat(U, ref_fe, conns, E)
  props_el = _element_level_properties(props, E)
  K_el = zeros(SMatrix{NxNDof, NxNDof, Float64, NxNDof * NxNDof})

  for q in 1:num_quadrature_points(ref_fe)
    interps = ref_fe.cell_interps.vals[q]
    state_old_q = _quadrature_level_state(state_old, q, E)
    K_q, state_new_q = func(physics, interps, u_el, x_el, state_old_q, props_el, t, dt)
    K_el = K_el + K_q
    for s in 1:length(state_old)
      state_new[s, q, E] = state_new_q[s]
    end
  end

  block_size = values(pattern.block_sizes)[block_id]
  block_offset = values(pattern.block_offsets)[block_id]
  start_id = (block_id - 1) * block_size + 
             (E - 1) * block_offset + 1
  end_id = start_id + block_offset - 1
  ids = start_id:end_id
  for (i, id) in enumerate(ids)
    Atomix.@atomic field[id] += K_el.data[i]
  end
end

"""
$(TYPEDSIGNATURES)
Assembly method for a block labelled as block_id. This is a GPU agnostic implementation
using KernelAbstractions and Atomix for eliminating race conditions

TODO add state variables and physics properties
"""
function _assemble_block_matrix!(
  field::F1, pattern::Patt, physics::Phys, ref_fe::R,
  U::F2, X::F3, state_old::S, state_new::S, props::P, t::T, dt::T,
  conns::C, block_id::Int, func::Func, backend::KA.Backend
) where {
  C    <: Connectivity,
  F1   <: AbstractArray{<:Number, 1},
  F2   <: AbstractField,
  F3   <: AbstractField,
  Func <: Function,
  P    <: Union{<:SVector, <:L2ElementField},
  Patt <: SparsityPattern,
  Phys <: AbstractPhysics,
  R    <: ReferenceFE,
  S    <: L2QuadratureField,
  T    <: Number
}
  kernel! = _assemble_block_matrix_kernel!(backend)
  kernel!(
    field, pattern, physics, ref_fe,
    U, X, state_old, state_new, props, t, dt,
    conns, block_id, func, ndrange=size(conns, 2)
  )
  return nothing
end
