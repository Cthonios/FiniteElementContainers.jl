function assemble_scalar_enzyme_safe!(
  storage::L2Field, pattern, dof,
  func::F, Uu, p,
  return_type::AssembledReturnType = AssembledScalar()
) where F <: Function
  fspace = function_space(dof)
  X = coordinates(p)
  t = current_time(p)
  Δt = time_step(p)
  U = p.field
  U_old = p.field_old
  _update_for_assembly!(p, dof, Uu)
  conns = fspace.elem_conns
  for (b, (
    block_physics, ref_fe
  )) in enumerate(zip(
    values(p.physics), values(fspace.ref_fes)
  ))
    _assemble_scalar_block_enzyme_safe!(
      KA.CPU(),
      block_view(storage, b),
      conns, 
      func,
      b,
      block_physics, ref_fe,
      X, t, Δt,
      U, U_old,
      p.state_old, p.state_new, p.properties,
      return_type
    )
  end
end

function _assemble_scalar_block_enzyme_safe!(
  ::KA.CPU,
  field,
  conns_all,
  func::Function,
  b::Int,
  physics::AbstractPhysics, ref_fe::ReferenceFE,
  X::AbstractField, t::T, dt::T,
  U::Solution, U_old::Solution, 
  state_old::StateVariableField, state_new::StateVariableField, props::PropertyField,
  return_type::R
) where {
  T        <: Number,
  Solution <: AbstractField,
  R        <: AssembledReturnType
}

  conns = conns_all.data
  coffset = conns_all.offsets[b]
  for e in 1:conns_all.nelems[b]
    conn = connectivity(ref_fe, conns, e, coffset)
    x_el, u_el, u_el_old = element_level_fields(ref_fe, conn, X, U, U_old)
    props_el = properties(props, e, b)

    for q in 1:num_cell_quadrature_points(ref_fe)
      interps = _cell_interpolants(ref_fe, q)
      state_old_q = state_variables(state_old, q, e, b)
      state_new_q = state_variables(state_new, q, e, b)
      val_q = func(physics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el)
      field[1, q, e] = val_q
    end
  end
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function assemble_vector_enzyme_safe!(
  assembler, func::F, Uu, p
) where F <: Function
  assemble_vector!(
    assembler.residual_storage, 
    assembler.vector_pattern, assembler.dof,
    func, Uu, p
  )
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function assemble_vector_enzyme_safe!(
  storage::AbstractField, pattern, dof, func::F, Uu, p
) where F <: Function
  fill!(storage, zero(eltype(storage)))
  fspace = function_space(dof)
  X = coordinates(p)
  t = current_time(p)
  Δt = time_step(p)
  U = p.field
  U_old = p.field_old
  _update_for_assembly!(p, dof, Uu)
  # return_type = AssembledVector()
  conns = fspace.elem_conns
  for (b, (
    block_physics, ref_fe
  )) in enumerate(zip(
    values(p.physics), values(fspace.ref_fes)
  ))
    _assemble_vector_block_enzyme_safe!(
      KA.CPU(),
      storage,
      conns,
      func,
      b,
      block_physics, ref_fe,
      X, t, Δt,
      U, U_old,
      p.state_old, p.state_new, p.properties
    )
  end
  
  return nothing
end

# CPU implementation
"""
$(TYPEDSIGNATURES)
Assembly method for a block labelled as block_id. This is a CPU implementation
with no threading.

TODO add state variables and physics properties
"""
function _assemble_vector_block_enzyme_safe!(
  ::KA.CPU,
  field,
  conns_all,
  func::Function,
  b::Int,
  physics::AbstractPhysics, ref_fe::ReferenceFE,
  X::AbstractField, t::T, dt::T,
  U::Solution, U_old::Solution, 
  state_old::StateVariableField, state_new::StateVariableField, props::PropertyField,
) where {
  T        <: Number,
  Solution <: AbstractField
}
  conns = conns_all.data
  coffset = conns_all.offsets[b]
  for e in 1:conns_all.nelems[b]
    conn = connectivity(ref_fe, conns, e, coffset)
    x_el, u_el, u_el_old = element_level_fields(ref_fe, conn, X, U, U_old)
    props_el = properties(props, e, b)

    for q in 1:num_cell_quadrature_points(ref_fe)
      interps = _cell_interpolants(ref_fe, q)
      state_old_q = state_variables(state_old, q, e, b)
      state_new_q = state_variables(state_new, q, e, b)
      # val_q = func(physics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el)
      # val_el = _accumulate_q_value(return_type, field, val_q, val_el, q, e)

      # hardcoded to inplace methods
      func(
        field, e, physics, t, dt, 
        props_el, state_old_q, state_new_q,
        conn, interps, x_el, u_el, u_el_old
      )
    end
  #   # _assemble_element!(field, val_el, conn, e)

  #   # writing inline to avoid atomic call
  #   # n_dofs = size(field, 1)
  #   # for d in axes(field, 1)
  #   #   for n in axes(conn, 1)
  #   #     global_id = n_dofs * (conn[n] - 1) + d
  #   #     local_id = n_dofs * (n - 1) + d
  #   #     field.data[global_id] += val_el[local_id]
  #   #   end
  #   # end
  end
  return nothing
end

<<<<<<< HEAD
# # GPU implementation
# # COV_EXCL_START
# KA.@kernel function _assemble_block_enzyme_safe_kernel!(
#   field,
#   conns::Conn, coffset::Int,
#   func::Function,
#   physics::AbstractPhysics, ref_fe::ReferenceFE,
#   X::AbstractField, t::T, dt::T,
#   U::Solution, U_old::Solution, 
#   state_old::S, state_new::S, props::AbstractArray,
#   return_type::R
# ) where {
#   T        <: Number,
#   Conn     <: AbstractArray,
#   Solution <: AbstractField,
#   S,       #<: L2QuadratureField
#   R        <: AssembledReturnType
# }
#   E = KA.@index(Global)
#   conn = connectivity(ref_fe, conns, E, coffset)
#   x_el, u_el, u_el_old = element_level_fields(ref_fe, conn, X, U, U_old)
#   props_el = _element_level_properties(props, E)
#   val_el = _element_scratch(return_type, ref_fe, U)
#   for q in 1:num_cell_quadrature_points(ref_fe)
#     interps = _cell_interpolants(ref_fe, q)
#     state_old_q = _quadrature_level_state(state_old, q, E)
#     state_new_q = _quadrature_level_state(state_new, q, E)
#     val_q = func(physics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el)
#     val_el = _accumulate_q_value(return_type, field, val_q, val_el, q, E)
#   end
=======
# GPU implementation
# COV_EXCL_START
KA.@kernel function _assemble_block_enzyme_safe_kernel!(
  field,
  conns::Conn, coffset::Int,
  func::Function,
  physics::AbstractPhysics, ref_fe::ReferenceFE,
  X::AbstractField, t::T, dt::T,
  U::Solution, U_old::Solution, 
  state_old::S, state_new::S, props::AbstractArray,
  return_type::R
) where {
  T        <: Number,
  Conn     <: AbstractArray,
  Solution <: AbstractField,
  S,       #<: L2QuadratureField
  R        <: AssembledReturnType
}
  E = KA.@index(Global)
  conn = connectivity(ref_fe, conns, E, coffset)
  x_el, u_el, u_el_old = element_level_fields(ref_fe, conn, E, X, U, U_old)
  props_el = _element_level_properties(props, E)
  val_el = _element_scratch(return_type, ref_fe, U)
  for q in 1:num_cell_quadrature_points(ref_fe)
    interps = _cell_interpolants(ref_fe, q)
    state_old_q = _quadrature_level_state(state_old, q, E)
    state_new_q = _quadrature_level_state(state_new, q, E)
    val_q = func(physics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el)
    val_el = _accumulate_q_value(return_type, field, val_q, val_el, q, E)
  end
>>>>>>> 115c6fc (Some work towards mixed space assembly. Its really hacky right now and using maximum code re-use. Definitely not efficient are ready for prime time though.)

#   # need the atomic here
#   _assemble_element!(field, val_el, conn, E)
# end
# # COV_EXCL_STOP

# # method for kernel generation
# function _assemble_block_enzyme_safe!(
#   backend::KA.Backend, 
#   field, 
#   conns, coffset::Int,
#   func,
#   physics, ref_fe,
#   X, t, dt, U, U_old, state_old, state_new, props,
#   return_type
# )
#   kernel! = _assemble_block_enzyme_safe_kernel!(backend)
#   kernel!(
#     field,
#     conns, coffset,
#     func,
#     physics, ref_fe,
#     X, t, dt,
#     U, U_old, 
#     state_old, state_new, props,
#     return_type,
#     ndrange = size(state_old, 3)
#   )
#   return nothing
# end
