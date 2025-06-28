# Top level method
function assemble!(assembler, Uu, p, ::Val{:residual_and_stiffness}, ::Type{H1Field})
  fill!(assembler.residual_storage, zero(eltype(assembler.residual_storage)))
  fill!(assembler.stiffness_storage, zero(eltype(assembler.stiffness_storage)))
  fspace = function_space(assembler, H1Field)
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
    _assemble_block_matrix_and_vector!(
      assembler.residual_storage, assembler.stiffness_storage, assembler.pattern, 
      block_physics, ref_fe, 
      p.h1_field, p.h1_coords, state_old, state_new, props, t, dt,
      conns, b, residual, stiffness,
      backend
    )
    KA.synchronize(backend)
  end
end

# CPU implementation

"""
$(TYPEDSIGNATURES)
Assembly method for a block labelled as block_id. This is a CPU implementation
with no threading.

TODO add state variables and physics properties
TODO remove Float64 typing below for eventual unitful use
"""
function _assemble_block_matrix_and_vector!(
  # assembler, physics, ref_fe, 
  residual_field::F1, stiffness_field::F2, pattern::Patt,
  physics::Phys, ref_fe::R,
  U::F3, X::F4, state_old::S, state_new::S, props::P, t::T, dt::T,
  conns::C, block_id::Int, 
  residual_func::Func1, stiffness_func::Func2, ::KA.CPU
) where {
  C     <: Connectivity,
  F1    <: AbstractField,
  F2    <: AbstractArray{<:Number, 1},
  F3    <: AbstractField,
  F4    <: AbstractField,
  Func1 <: Function,
  Func2 <: Function,
  P     <: Union{<:SVector, <:L2ElementField},
  Patt  <: SparsityPattern,
  Phys  <: AbstractPhysics,
  R     <: ReferenceFE,
  S     <: L2QuadratureField,
  T     <: Number
}
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
  for e in axes(conns, 2)
    x_el = _element_level_fields_flat(X, ref_fe, conns, e)
    u_el = _element_level_fields_flat(U, ref_fe, conns, e)
    props_el = _element_level_properties(props, e)
    R_el = zeros(SVector{NxNDof, eltype(residual_field)})
    K_el = zeros(SMatrix{NxNDof, NxNDof, eltype(stiffness_field), NxNDof * NxNDof})

    for q in 1:num_quadrature_points(ref_fe)
      interps = _cell_interpolants(ref_fe, q)
      state_old_q = _quadrature_level_state(state_old, q, e)
      R_q, state_new_q = residual_func(physics, interps, u_el, x_el, state_old_q, props_el, t, dt)
      K_q, state_new_q = stiffness_func(physics, interps, u_el, x_el, state_old_q, props_el, t, dt)
      R_el = R_el + R_q
      K_el = K_el + K_q
      for s in 1:length(state_old)
        state_new[s, q, e] = state_new_q[s]
      end
    end
    
    @views _assemble_element!(residual_field, R_el, conns[:, e], e, block_id)
    @views _assemble_element!(pattern, stiffness_field, K_el, e, block_id)
  end
  return nothing
end

# TODO implement GPU version
#= @coverage-ignore =#
KA.@kernel function _assemble_block_matrix_and_vector_kernel!(
  residual_field::F1, stiffness_field::F2, pattern::Patt, 
  physics::Phys, ref_fe::R,
  U::F3, X::F4, state_old::S, state_new::S, props::P, t::T, dt::T,
  conns::C, block_id::Int, residual_func::Func1, stiffness_func::Func2
) where {
  C     <: Connectivity,
  F1    <: AbstractField,
  F2    <: AbstractArray{<:Number, 1},
  F3    <: AbstractField,
  F4    <: AbstractField,
  Func1 <: Function,
  Func2 <: Function,
  P     <: Union{<:SVector, <:L2ElementField},
  Patt  <: SparsityPattern,
  Phys  <: AbstractPhysics,
  R     <: ReferenceFE,
  S     <: L2QuadratureField,
  T     <: Number
}
  E = KA.@index(Global)

  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND

  x_el = _element_level_fields_flat(X, ref_fe, conns, E)
  u_el = _element_level_fields_flat(U, ref_fe, conns, E)
  props_el = _element_level_properties(props, E)
  R_el = zeros(SVector{NxNDof, eltype(residual_field)})
  K_el = zeros(SMatrix{NxNDof, NxNDof, eltype(stiffness_field), NxNDof * NxNDof})

  for q in 1:num_quadrature_points(ref_fe)
    interps = _cell_interpolants(ref_fe, q)
    state_old_q = _quadrature_level_state(state_old, q, E)
    R_q, state_new_q = residual_func(physics, interps, u_el, x_el, state_old_q, props_el, t, dt)
    K_q, state_new_q = stiffness_func(physics, interps, u_el, x_el, state_old_q, props_el, t, dt)
    R_el = R_el + R_q
    K_el = K_el + K_q
    for s in 1:length(state_old)
      state_new[s, q, E] = state_new_q[s]
    end
  end

  # now assemble atomically
  n_dofs = size(residual_field, 1)
  for i in 1:size(conns, 1)
    for d in 1:n_dofs
      global_id = n_dofs * (conns[i, E] - 1) + d
      local_id = n_dofs * (i - 1) + d
      Atomix.@atomic residual_field.vals[global_id] += R_el[local_id]
    end
  end

  block_size = values(pattern.block_sizes)[block_id]
  block_offset = values(pattern.block_offsets)[block_id]
  start_id = (block_id - 1) * block_size + 
             (E - 1) * block_offset + 1
  end_id = start_id + block_offset - 1
  ids = start_id:end_id
  for (i, id) in enumerate(ids)
    Atomix.@atomic stiffness_field[id] += K_el.data[i]
  end
end

# """
# $(TYPEDSIGNATURES)
# Assembly method for a block labelled as block_id. This is a GPU agnostic implementation
# using KernelAbstractions and Atomix for eliminating race conditions

# TODO add state variables and physics properties
# """
function _assemble_block_matrix_and_vector!(
  residual_field::F1, stiffness_field::F2, pattern::Patt, 
  physics::Phys, ref_fe::R,
  U::F3, X::F4, state_old::S, state_new::S, props::P, t::T, dt::T,
  conns::C, block_id::Int, 
  residual_func::Func1, stiffness_func::Func2, 
  backend::KA.Backend
) where {
  C     <: Connectivity,
  F1    <: AbstractField,
  F2    <: AbstractArray{<:Number, 1},
  F3    <: AbstractField,
  F4    <: AbstractField,
  Func1 <: Function,
  Func2 <: Function,
  P     <: Union{<:SVector, <:L2ElementField},
  Patt  <: SparsityPattern,
  Phys  <: AbstractPhysics,
  R     <: ReferenceFE,
  S     <: L2QuadratureField,
  T     <: Number
}
  kernel! = _assemble_block_matrix_and_vector_kernel!(backend)
  kernel!(
    residual_field, stiffness_field, pattern, 
    physics, ref_fe,
    U, X, state_old, state_new, props, t, dt,
    conns, block_id, 
    residual_func, stiffness_func,
    ndrange=size(conns, 2)
  )
  return nothing
end
