function assemble_mass!(
  assembler, func::F, Uu, p
) where F <: Function
  assemble_matrix!(
    assembler.mass_storage, assembler.pattern, assembler.dof,
    func, Uu, p
  )
end

function assemble_stiffness!(
  assembler, func::F, Uu, p
) where F <: Function
  assemble_matrix!(
    assembler.stiffness_storage, assembler.pattern, assembler.dof,
    func, Uu, p
  )
end

"""
$(TYPEDSIGNATURES)
Note this is hard coded to storing the assembled sparse matrix in 
the stiffness_storage field of assembler.
"""
function assemble_matrix!(
  # assembler, func::F, Uu, p, ::Type{H1Field};
  # storage_sym=Val{:stiffness_storage}()
  storage, pattern, dof, func::F, Uu, p
) where F <: Function
  # storage = getfield(assembler, storage_sym)
  # storage = _get_storage(assembler, storage_sym)
  fill!(storage, zero(eltype(storage)))
  fspace = function_space(dof)
  t = current_time(p.times)
  dt = time_step(p.times)
  _update_for_assembly!(p, dof, Uu)
  for (b, (conns, block_physics, state_old, state_new, props)) in enumerate(zip(
    values(fspace.elem_conns), 
    values(p.physics),
    values(p.state_old), values(p.state_new),
    values(p.properties)
  ))
    ref_fe = values(fspace.ref_fes)[b]
    # TODO re-enable back-end checking
    # backend = _check_backends(assembler, p.h1_field, p.h1_coords, state_old, state_new, conns)
    backend = KA.get_backend(p.h1_field)
    _assemble_block_matrix!(
      storage, pattern, block_physics, ref_fe,
      p.h1_field, p.h1_field_old, p.h1_coords, state_old, state_new, props, t, dt,
      conns, b, func,
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
  U::F2, U_old::F2, X::F3, state_old::S, state_new::S, props::P, t::T, dt::T,
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

  for e in axes(conns, 2)
    x_el = _element_level_fields_flat(X, ref_fe, conns, e)
    u_el = _element_level_fields_flat(U, ref_fe, conns, e)
    u_el_old = _element_level_fields_flat(U_old, ref_fe, conns, e)
    props_el = _element_level_properties(props, e)
    K_el = _element_scratch_matrix(ref_fe, U)

    for q in 1:num_quadrature_points(ref_fe)
      interps = _cell_interpolants(ref_fe, q)
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
# COV_EXCL_START
KA.@kernel function _assemble_block_matrix_kernel!(
  field::F1, pattern::Patt, physics::Phys, ref_fe::R,
  U::F2, U_old::F2, X::F3, state_old::S, state_new::S, props::P, t::T, dt::T,
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

  x_el = _element_level_fields_flat(X, ref_fe, conns, E)
  u_el = _element_level_fields_flat(U, ref_fe, conns, E)
  u_el_old = _element_level_fields_flat(U_old, ref_fe, conns, E)
  props_el = _element_level_properties(props, E)
  K_el = _element_scratch_matrix(ref_fe, U)
  for q in 1:num_quadrature_points(ref_fe)
    interps = _cell_interpolants(ref_fe, q)
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
# COV_EXCL_STOP

"""
$(TYPEDSIGNATURES)
Assembly method for a block labelled as block_id. This is a GPU agnostic implementation
using KernelAbstractions and Atomix for eliminating race conditions

TODO add state variables and physics properties
"""
function _assemble_block_matrix!(
  field::F1, pattern::Patt, physics::Phys, ref_fe::R,
  U::F2, U_old::F2, X::F3, state_old::S, state_new::S, props::P, t::T, dt::T,
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
    U, U_old, X, state_old, state_new, props, t, dt,
    conns, block_id, func, ndrange=size(conns, 2)
  )
  return nothing
end
