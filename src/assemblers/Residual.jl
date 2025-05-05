# Top level method

function assemble!(assembler, ::Type{H1Field}, p, val_sym::Val{:residual})
  fspace = assembler.dof.H1_vars[1].fspace
  dt = time_step(p.times)
  _zero_storage(assembler, val_sym)
  for (b, (conns, block_physics, state_old, state_new, props)) in enumerate(zip(
    values(fspace.elem_conns), 
    values(p.physics),
    values(p.state_old), values(p.state_new),
    values(p.properties)
  ))
    ref_fe = values(fspace.ref_fes)[b]
    backend = _check_backends(assembler, p.h1_field, p.h1_coords, state_old, state_new, conns)
    _assemble_block_residual!(
      assembler, block_physics, ref_fe, 
      p.h1_field, p.h1_coords, state_old, state_new, props, dt,
      conns, b, 
      backend
    )
  end
end


# CPU Implementation

"""
$(TYPEDSIGNATURES)
Assembly method for a block labelled as block_id. This is a CPU implementation
with no threading.

TODO add state variables and physics properties
TODO remove Float64 typing below for eventual unitful use
"""
function _assemble_block_residual!(
  assembler, physics, ref_fe, 
  U, X, state_old, state_new, props, dt,
  conns, block_id, ::KA.CPU
)
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
  for e in axes(conns, 2)
    x_el = _element_level_fields(X, ref_fe, conns, e)
    u_el = _element_level_fields(U, ref_fe, conns, e)
    props_el = _element_level_properties(props, e)
    R_el = zeros(SVector{NxNDof, eltype(assembler.residual_storage)})

    for q in 1:num_quadrature_points(ref_fe)
      interps = MappedInterpolants(ref_fe.cell_interps.vals[q], x_el)
      state_old_q = _quadrature_level_state(state_old, q, e)
      R_q = residual(physics, interps, u_el, state_old_q, props_el, dt)
      R_el = R_el + R_q
    end
    
    @views _assemble_element!(assembler.residual_storage, R_el, conns[:, e], e, block_id)
  end
end

# TODO hardcoded to H1 fields right now.
"""
$(TYPEDSIGNATURES)
"""
function _residual(asm::AbstractAssembler, ::KA.CPU)
  # for n in axes(asm.residual_unknowns, 1)
  #   asm.residual_unknowns[n] = asm.residual_storage[asm.dof.H1_unknown_dofs[n]]
  # end
  @views asm.residual_unknowns .= asm.residual_storage[asm.dof.H1_unknown_dofs]
  return asm.residual_unknowns
end

# GPU implementation

"""
$(TYPEDSIGNATURES)
Kernel for residual block assembly

TODO mark const fields
"""
KA.@kernel function _assemble_block_residual_kernel!(
  assembler, physics, ref_fe, 
  U, X, state_old, state_new, props, dt,
  conns, block_id
)
  E = KA.@index(Global)

  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND

  x_el = _element_level_fields(X, ref_fe, conns, E)
  u_el = _element_level_fields(U, ref_fe, conns, E)
  props_el = _element_level_properties(props, E)
  R_el = zeros(SVector{NxNDof, Float64})

  for q in 1:num_quadrature_points(ref_fe)
    interps = MappedInterpolants(ref_fe.cell_interps.vals[q], x_el)
    state_old_q = _quadrature_level_state(state_old, q, E)
    R_q = residual(physics, interps, u_el, state_old_q, props_el, dt)
    R_el = R_el + R_q
  end

  # now assemble atomically
  n_dofs = size(assembler.residual_storage, 1)
  for i in 1:size(conns, 1)
    for d in 1:n_dofs
      global_id = n_dofs * (conns[i, E] - 1) + d
      local_id = n_dofs * (i - 1) + d
      Atomix.@atomic assembler.residual_storage.vals[global_id] += R_el[local_id]
    end
  end
end

"""
$(TYPEDSIGNATURES)
Assembly method for a block labelled as block_id. This is a GPU agnostic implementation
using KernelAbstractions and Atomix for eliminating race conditions

TODO add state variables and physics properties
"""
function _assemble_block_residual!(
  assembler, physics, ref_fe, 
  U, X, state_old, state_new, props, dt,
  conns, block_id, backend::KA.Backend
)
  kernel! = _assemble_block_residual_kernel!(backend)
  kernel!(
    assembler, physics, ref_fe, 
    U, X, state_old, state_new, props, dt,
    conns, block_id, ndrange=size(conns, 2)
  )
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
KA.@kernel function _extract_residual_unknowns!(Ru, unknown_dofs, R)
  N = KA.@index(Global)
  Ru[N] = R[unknown_dofs[N]]
end

"""
$(TYPEDSIGNATURES)
"""
function _residual(asm::AbstractAssembler, backend::KA.Backend)
  kernel! = _extract_residual_unknowns!(backend)
  kernel!(asm.residual_unknowns, 
          asm.dof.H1_unknown_dofs, 
          asm.residual_storage, 
          ndrange=length(asm.dof.H1_unknown_dofs))
  return asm.residual_unknowns
end 
