function assemble_mass!(
  assembler, func::F, Uu, p
) where F <: Function
  assemble_matrix!(
    assembler.mass_storage, assembler.matrix_pattern, assembler.dof,
    func, Uu, p
  )
end

function assemble_stiffness!(
  assembler, func::F, Uu, p
) where F <: Function
  assemble_matrix!(
    assembler.stiffness_storage, assembler.matrix_pattern, assembler.dof,
    func, Uu, p
  )
end

"""
$(TYPEDSIGNATURES)
Note this is hard coded to storing the assembled sparse matrix in 
the stiffness_storage field of assembler.
"""
function assemble_matrix!(
  storage, pattern, dof, func::F, Uu, p
) where F <: Function
  fill!(storage, zero(eltype(storage)))
  fspace = function_space(dof)
  t = current_time(p.times)
  dt = time_step(p.times)
  _update_for_assembly!(p, dof, Uu)
  for (
    conns, block_start_index, block_el_level_size,
    block_physics, ref_fe, 
    state_old, state_new, props
  ) in zip(
    values(fspace.elem_conns), 
    values(pattern.block_start_indices),
    values(pattern.block_el_level_sizes),
    values(p.physics), values(fspace.ref_fes),
    values(p.state_old), values(p.state_new),
    values(p.properties),
  )
    # TODO re-enable back-end checking
    backend = KA.get_backend(p.h1_field)
    _assemble_block_matrix!(
      backend,
      storage,
      conns, block_start_index, block_el_level_size,
      func,
      block_physics, ref_fe,
      p.h1_coords, t, dt,
      p.h1_field, p.h1_field_old, 
      state_old, state_new, props
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
  ::KA.CPU,
  field::AbstractArray{<:Number, 1}, 
  conns::Connectivity, block_start_index::Int, block_el_level_size::Int,
  func::Function,
  physics::AbstractPhysics, ref_fe::ReferenceFE,
  X::AbstractField, t::T, dt::T,
  U::Solution, U_old::Solution, 
  state_old::S, state_new::S, props::AbstractArray
) where {
  T        <: Number,
  Solution <: AbstractField,
  S        #<: L2QuadratureField
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
      state_new_q = _quadrature_level_state(state_new, q, e)
      K_q = func(physics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el)
      K_el = K_el + K_q
    end
    _assemble_element!(field, K_el, e, block_start_index, block_el_level_size)
  end
  return nothing
end


# GPU implementation
# COV_EXCL_START
KA.@kernel function _assemble_block_matrix_kernel!(
  field::AbstractArray{<:Number, 1}, 
  conns::Connectivity, block_start_index::Int, block_el_level_size::Int,
  func::Function,
  physics::AbstractPhysics, ref_fe::ReferenceFE,
  X::AbstractField, t::T, dt::T,
  U::Solution, U_old::Solution, 
  state_old::S, state_new::S, props::AbstractArray
) where {
  T        <: Number,
  Solution <: AbstractField,
  S        #<: L2QuadratureField
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
    state_new_q = _quadrature_level_state(state_new, q, E)
    K_q = func(physics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el)
    K_el = K_el + K_q
  end

  # leaving here just in case
  # block_start_index = values(pattern.block_start_indices)[block_id]
  # block_el_level_size = values(pattern.block_el_level_sizes)[block_id]
  # start_id = block_start_index + 
  #            (E - 1) * block_el_level_size
  # end_id = start_id + block_el_level_size - 1
  # ids = start_id:end_id
  # for (i, id) in enumerate(ids)
  #   Atomix.@atomic field[id] += K_el.data[i]
  # end
  _assemble_element!(field, K_el, E, block_start_index, block_el_level_size)
end
# COV_EXCL_STOP

"""
$(TYPEDSIGNATURES)
Assembly method for a block labelled as block_id. This is a GPU agnostic implementation
using KernelAbstractions and Atomix for eliminating race conditions

TODO add state variables and physics properties
"""
function _assemble_block_matrix!(
  backend::KA.Backend,
  field::AbstractArray{<:Number, 1}, 
  conns::Connectivity, block_start_index::Int, block_el_level_size::Int,
  func::Function,
  physics::AbstractPhysics, ref_fe::ReferenceFE,
  X::AbstractField, t::T, dt::T,
  U::Solution, U_old::Solution, 
  state_old::S, state_new::S, props::AbstractArray
) where {
  T        <: Number,
  Solution <: AbstractField,
  S        #<: L2QuadratureField
}
  kernel! = _assemble_block_matrix_kernel!(backend)
  kernel!(
    field,
    conns, block_start_index, block_el_level_size, 
    func,
    physics, ref_fe,
    X, t, dt,
    U, U_old, 
    state_old, state_new, props,
    ndrange=size(conns, 2)
  )
  return nothing
end
