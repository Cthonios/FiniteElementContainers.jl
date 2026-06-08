"""
$(TYPEDSIGNATURES)
Matrix-free action assembly. `func_action` has signature
  func_action(physics, interps, x_el, t, Δt, u_el, u_el_old, v_el,
              state_old_q, state_new_q, props_el) → SVector
and returns the element-level product K_q·v_el directly, avoiding
formation of the full element stiffness/mass matrix.
"""
function assemble_matrix_free_action!(
  assembler, func_action::F, Uu, Vu, p
) where F <: Function
  assemble_matrix_free_action!(
    assembler.stiffness_action_storage,
    assembler.vector_pattern, assembler.dof,
    func_action, Uu, Vu, p
  )
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function assemble_matrix_free_action!(
  storage, pattern, dof, func_action, Uu, Vu, p
)
  fill!(storage, zero(eltype(storage)))
  fspace = function_space(dof)
  X = coordinates(p)
  t = current_time(p)
  Δt = time_step(p)
  U = p.field
  U_old = p.field_old
  V = p.hvp_scratch_field
  _update_for_assembly!(p, dof, Uu, Vu)
  conns = fspace.elem_conns
  foreach_block(fspace, p) do physics, props, ref_fe, b
    _assemble_block_matrix_free_action!(
      storage,
      conns.data, conns.offsets[b],
      func_action,
      physics, ref_fe,
      X, t, Δt,
      U, U_old, V,
      block_view(p.state_old, b), block_view(p.state_new, b), props
    )
  end
end

function _assemble_block_matrix_free_action!(
  field::AbstractField,
  conns::Conn, coffset,
  func_action::Function,
  physics::AbstractPhysics, ref_fe::ReferenceFE,
  X::AbstractField, t::T, Δt::T,
  U::Solution, U_old::Solution, V::Solution,
  state_old::S, state_new::S, props::AbstractArray
) where {
  T        <: Number,
  Conn     <: AbstractArray,
  Solution <: AbstractField,
  S
}
  fec_foraxes(state_old, 3) do e
    conn = connectivity(ref_fe, conns, e, coffset)
    x_el, u_el, u_el_old, v_el = element_level_fields(ref_fe, conn, X, U, U_old, V)
    props_el = _element_level_properties(props, e)
    Kv_el = _element_scratch(AssembledVector(), ref_fe, U)
    for q in 1:num_cell_quadrature_points(ref_fe)
      interps = _cell_interpolants(ref_fe, q)
      state_old_q = _quadrature_level_state(state_old, q, e)
      state_new_q = _quadrature_level_state(state_new, q, e)
      Kv_q = func_action(physics, interps, x_el, t, Δt, u_el, u_el_old, v_el, state_old_q, state_new_q, props_el)
      Kv_el = Kv_el + Kv_q
    end
    _assemble_element!(field, Kv_el, conn, e)
  end
end

"""
$(TYPEDSIGNATURES)
Full-DOF variant of `assemble_matrix_free_action!`.  Distinct from the
free-DOF variant in that the BC slots of `v_full` are honored as part
of the action — used for residual-style evaluations where `v_full`
carries a real BC contribution (e.g., the Newmark inertial residual
`c_M * M * dU` with `dU_BC ≠ 0` when the Dirichlet BC is time-varying).

In the free-DOF variant `assemble_matrix_free_action!(..., Uu, Vu, p)`,
`Vu` is interpreted as a perturbation of the unknowns and the BC slots
of `p.hvp_scratch_field` are implicitly zero — correct for Krylov-style
Jacobian-action calls, where `v` ranges over unknowns only.

For this full-DOF variant the caller is responsible for assembling
`U_full = [Uu; U_BC]` and `v_full = [v_free; v_BC]` themselves; both
vectors must have length equal to the underlying field data, and the
field's Dirichlet slots are NOT overwritten from `bc_cache.vals` (doing
so would clobber the BC contribution the caller deliberately placed
in `U_full`).
"""
function assemble_matrix_free_action_full!(
  assembler, func_action::F, U_full, v_full, p
) where F <: Function
  assemble_matrix_free_action_full!(
    assembler.stiffness_action_storage,
    assembler.vector_pattern, assembler.dof,
    func_action, U_full, v_full, p
  )
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function assemble_matrix_free_action_full!(
  storage, pattern, dof, func_action, U_full, v_full, p
)
  fill!(storage, zero(eltype(storage)))
  fspace = function_space(dof)
  X = coordinates(p)
  t = current_time(p)
  Δt = time_step(p)
  U = p.field
  U_old = p.field_old
  V = p.hvp_scratch_field
  _update_for_assembly_full!(p, U_full, v_full)
  conns = fspace.elem_conns
  foreach_block(fspace, p) do physics, props, ref_fe, b
    _assemble_block_matrix_free_action!(
      storage,
      conns.data, conns.offsets[b],
      func_action,
      physics, ref_fe,
      X, t, Δt,
      U, U_old, V,
      block_view(p.state_old, b), block_view(p.state_new, b), props
    )
  end
  # The free-DOF entry point above relies on `p.hvp_scratch_field`'s BC
  # slots being zero (it never writes them, and the kernel reads them as
  # part of v_el).  Restore that invariant after we deliberately put
  # v_BC into the scratch — otherwise a subsequent Krylov J·v call that
  # comes through the free-DOF entry point silently gets contaminated
  # by leftover v_BC values.
  cache = p.dirichlet_bcs.bc_cache
  data = p.hvp_scratch_field.data
  Z = zero(eltype(data))
  fec_foreach(cache.dofs) do I
    data[cache.dofs[I]] = Z
  end
end

"""
$(TYPEDSIGNATURES)
"""
function assemble_matrix_action!(
  assembler, func::F, Uu, Vu, p
) where F <: Function
  assemble_matrix_action!(
    assembler.stiffness_action_storage,
    assembler.vector_pattern, assembler.dof,
    func, Uu, Vu, p;
    use_inplace_methods = _use_inplace_methods(assembler)
  )
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function assemble_matrix_action!(
  storage, pattern, dof, func, Uu, Vu, p;
  use_inplace_methods::Bool = false
)
  fill!(storage, zero(eltype(storage)))
  fspace = function_space(dof)
  X = coordinates(p)
  t = current_time(p)
  Δt = time_step(p)
  U = p.field
  U_old = p.field_old
  V = p.hvp_scratch_field
  _update_for_assembly!(p, dof, Uu, Vu)
  conns = fspace.elem_conns
  foreach_block(fspace, p) do physics, props, ref_fe, b
    if use_inplace_methods
      _assemble_block_matrix_action!(
        storage,
        func,
        physics,
        t, Δt,
        props,
        block_view(p.state_old, b), block_view(p.state_new, b),
        conns.data, conns.offsets[b], ref_fe, X, U, U_old, V
      )
    else
      _assemble_block_matrix_action!(
        storage,
        conns.data, conns.offsets[b],
        func,
        physics, ref_fe,
        X, t, Δt,
        U, U_old, V,
        block_view(p.state_old, b), block_view(p.state_new, b), props
      )
    end
  end
end

function _assemble_block_matrix_action!(
  field::AbstractField, 
  conns::Conn, coffset,
  func::Function,
  physics::AbstractPhysics, ref_fe::ReferenceFE,
  X::AbstractField, t::T, Δt::T,
  U::Solution, U_old::Solution, V::Solution,
  state_old::S, state_new::S, props::AbstractArray
) where {
  T        <: Number,
  Conn     <: AbstractArray,
  Solution <: AbstractField,
  S        #<: L2QuadratureField
}
  fec_foraxes(state_old, 3) do e
    conn = connectivity(ref_fe, conns, e, coffset)
    x_el, u_el, u_el_old, v_el = element_level_fields(ref_fe, conn, X, U, U_old, V)
    props_el = _element_level_properties(props, e)
    K_el = _element_scratch(AssembledMatrix(), ref_fe, U)
    for q in 1:num_cell_quadrature_points(ref_fe)
      interps = _cell_interpolants(ref_fe, q)
      state_old_q = _quadrature_level_state(state_old, q, e)
      state_new_q = _quadrature_level_state(state_new, q, e)
      K_q = func(physics, interps, x_el, t, Δt, u_el, u_el_old, state_old_q, state_new_q, props_el)
      K_el = K_el + K_q
    end
    Kv_el = K_el * v_el

    _assemble_element!(field, Kv_el, conn, e)
  end
end

function _assemble_block_matrix_action!(
  field,
  func!::Function,
  physics::AbstractPhysics,
  t::T, Δt::T,
  props::P, state_old::S, state_new::S,
  conns::Conn, coffset::Int, ref_fe::ReferenceFE,
  X::AbstractField, U::Solution, U_old::Solution, V::Solution
) where {
  T        <: Number,
  S,
  P,
  Conn     <: AbstractArray,
  Solution <: AbstractField
}
  fec_foraxes(state_old, 3) do e
    conn = connectivity(ref_fe, conns, e, coffset)
    x_el, u_el, u_el_old, v_el = element_level_fields(ref_fe, conn, X, U, U_old, V)
    props_el = _element_level_properties(props, e)
    for q in 1:num_cell_quadrature_points(ref_fe)
      interps = _cell_interpolants(ref_fe, q)
      state_old_q = _quadrature_level_state(state_old, q, e)
      state_new_q = _quadrature_level_state(state_new, q, e)
      func!(
        field, e, physics, t, Δt, 
        props_el, state_old_q, state_new_q,
        conn, interps, x_el, u_el, u_el_old, v_el
      )
    end
  end
end
