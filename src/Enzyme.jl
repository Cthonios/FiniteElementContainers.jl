# CPU implementation
"""
$(TYPEDSIGNATURES)
Assembly method for a block labelled as block_id. This is a CPU implementation
with no threading.

TODO add state variables and physics properties
"""
function _assemble_block_enzyme_safe!(
  ::KA.CPU,
  # field::AbstractArray{<:Number, 1}, 
  field,
#   conns::Conn, nelem::Int, coffset::Int,
  conns::Conn, coffset::Int,
  block_start_index::Int, block_el_level_size::Int,
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

  for e in axes(state_old, 3)
    conn = connectivity(ref_fe, conns, e, coffset)
    x_el = _element_level_fields_flat(X, ref_fe, conn)#s, e)
    u_el = _element_level_fields_flat(U, ref_fe, conn)#s, e)
    u_el_old = _element_level_fields_flat(U_old, ref_fe, conn)#s, e)
    props_el = _element_level_properties(props, e)
    val_el = _element_scratch(return_type, ref_fe, U)

    for q in 1:num_cell_quadrature_points(ref_fe)
      interps = _cell_interpolants(ref_fe, q)
      state_old_q = _quadrature_level_state(state_old, q, e)
      state_new_q = _quadrature_level_state(state_new, q, e)
      val_q = func(physics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el)
      val_el = _accumulate_q_value(return_type, field, val_q, val_el, q, e)
    end
    _assemble_element!(field, val_el, conn, e, block_start_index, block_el_level_size)
  end
  return nothing
end

# GPU implementation
# COV_EXCL_START
KA.@kernel function _assemble_block_enzyme_safe_kernel!(
  # field::AbstractArray{<:Number, 1}, 
  field,
  conns::Conn, coffset::Int,
  block_start_index::Int, block_el_level_size::Int,
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
  x_el = _element_level_fields_flat(X, ref_fe, conn)
  u_el = _element_level_fields_flat(U, ref_fe, conn)
  u_el_old = _element_level_fields_flat(U_old, ref_fe, conn)
  props_el = _element_level_properties(props, E)
  val_el = _element_scratch(return_type, ref_fe, U)
  for q in 1:num_cell_quadrature_points(ref_fe)
    interps = _cell_interpolants(ref_fe, q)
    state_old_q = _quadrature_level_state(state_old, q, E)
    state_new_q = _quadrature_level_state(state_new, q, E)
    val_q = func(physics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el)
    val_el = _accumulate_q_value(return_type, field, val_q, val_el, q, E)
  end

  _assemble_element!(field, val_el, conn, E, block_start_index, block_el_level_size)
end
# COV_EXCL_STOP

# method for kernel generation
function _assemble_block_enzyme_safe!(
  backend::KA.Backend, 
  field, 
#   conns, nelem, coffset::Int,
  conns, coffset::Int,
  block_start_index, block_el_level_size,
  func,
  physics, ref_fe,
  X, t, dt, U, U_old, state_old, state_new, props,
  return_type
)
  kernel! = _assemble_block_enzyme_safe_kernel!(backend)
  kernel!(
    field,
    conns, coffset,
    block_start_index, block_el_level_size, 
    func,
    physics, ref_fe,
    X, t, dt,
    U, U_old, 
    state_old, state_new, props,
    return_type,
    ndrange = size(state_old, 3)
  )
  return nothing
end

# COV_EXCL_START
KA.@kernel function _update_field_unknowns_enzyme_safe_kernel!(
    U::AbstractField, 
    # dof::DofManager{true, IT, IDs, Var}, 
    unknown_dofs::IDs,
    Uu::V,
    flag::Bool
) where {V <: AbstractVector{<:Number}, IDs}
    N = KA.@index(Global)
    if flag
        @inbounds U.data[unknown_dofs[N]] = Uu[unknown_dofs[N]]
    else
        @inbounds U.data[unknown_dofs[N]] = Uu[N]
    end
end
# COV_EXCL_STOP
  
function _update_field_unknowns_enzyme_safe!(
    U::AbstractField, 
    dof::DofManager{flag, IT, IDs, Var}, 
    Uu::T, 
    backend::KA.Backend
) where {
    T <: AbstractVector{<:Number},
    flag, IT, IDs, Var
}
    kernel! = _update_field_unknowns_enzyme_safe_kernel!(backend)
    kernel!(U, dof.unknown_dofs, Uu, flag, ndrange = length(dof.unknown_dofs))
    return nothing
end
  
# Need a seperate CPU method since CPU is basically busted in KA
function _update_field_unknowns_enzyme_safe!(
    U::AbstractField, 
    dof::DofManager{false, IT, IDs, Var}, 
    Uu::T, 
    ::KA.CPU
) where {T <: AbstractVector{<:Number}, IT, IDs, Var}
    U[dof.unknown_dofs] .= Uu
    return nothing
end

function _update_field_unknowns_enzyme_safe!(
    U::AbstractField, 
    dof::DofManager{true, IT, IDs, Var}, 
    Uu::T, 
    ::KA.CPU
) where {T <: AbstractVector{<:Number}, IT, IDs, Var}
    @views U[dof.unknown_dofs] .= Uu[dof.unknown_dofs]
    return nothing
end
