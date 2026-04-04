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
  storage, pattern, dof, func::F, Uu, p
) where F <: Function
  fill!(storage, zero(eltype(storage)))
  fspace = function_space(dof)
  X = coordinates(p)
  t = current_time(p)
  Δt = time_step(p)
  U = p.field
  U_old = p.field_old
  _update_for_assembly!(p, dof, Uu, true)
  return_type = AssembledVector()
  conns = fspace.elem_conns
  for (b, (
    block_physics, ref_fe, props
  )) in enumerate(zip(
    values(p.physics), values(fspace.ref_fes),
    values(p.properties)
  ))
    _assemble_block_enzyme_safe!(
      KA.get_backend(storage),
      storage,
      conns.data, conns.offsets[b], 
      func,
      block_physics, ref_fe,
      X, t, Δt,
      U, U_old,
      block_view(p.state_old, b), block_view(p.state_new, b), props,
      return_type
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
function _assemble_block_enzyme_safe!(
  ::KA.CPU,
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

  for e in axes(state_old, 3)
    conn = connectivity(ref_fe, conns, e, coffset)
    x_el, u_el, u_el_old = element_level_fields(ref_fe, conn, X, U, U_old)
    props_el = _element_level_properties(props, e)
    val_el = _element_scratch(return_type, ref_fe, U)

    for q in 1:num_cell_quadrature_points(ref_fe)
      interps = _cell_interpolants(ref_fe, q)
      state_old_q = _quadrature_level_state(state_old, q, e)
      state_new_q = _quadrature_level_state(state_new, q, e)
      val_q = func(physics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el)
      val_el = _accumulate_q_value(return_type, field, val_q, val_el, q, e)
    end
    _assemble_element!(field, val_el, conn, e)
  end
  return nothing
end

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
  x_el, u_el, u_el_old = element_level_fields(ref_fe, conn, X, U, U_old)
  props_el = _element_level_properties(props, E)
  val_el = _element_scratch(return_type, ref_fe, U)
  for q in 1:num_cell_quadrature_points(ref_fe)
    interps = _cell_interpolants(ref_fe, q)
    state_old_q = _quadrature_level_state(state_old, q, E)
    state_new_q = _quadrature_level_state(state_new, q, E)
    val_q = func(physics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el)
    val_el = _accumulate_q_value(return_type, field, val_q, val_el, q, E)
  end

  _assemble_element!(field, val_el, conn, E)
end
# COV_EXCL_STOP

# method for kernel generation
function _assemble_block_enzyme_safe!(
  backend::KA.Backend, 
  field, 
  conns, coffset::Int,
  func,
  physics, ref_fe,
  X, t, dt, U, U_old, state_old, state_new, props,
  return_type
)
  kernel! = _assemble_block_enzyme_safe_kernel!(backend)
  kernel!(
    field,
    conns, coffset,
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
