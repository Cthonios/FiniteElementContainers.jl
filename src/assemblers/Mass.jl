# top level method

function assemble!(assembler, ::Type{H1Field}, p, val_sym::Val{:mass})
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
    _assemble_block_mass!(
      assembler, block_physics, ref_fe, 
      p.h1_field, p.h1_coords, state_old, state_new, props, dt,
      conns, b, 
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
function _assemble_block_mass!(
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
    K_el = zeros(SMatrix{NxNDof, NxNDof, eltype(assembler.mass_storage), NxNDof * NxNDof})

    for q in 1:num_quadrature_points(ref_fe)
      interps = ref_fe.cell_interps.vals[q]
      state_old_q = _quadrature_level_state(state_old, q, e)
      K_q, state_new_q = mass(physics, interps, u_el, x_el, state_old_q, props_el, dt)
      K_el = K_el + K_q
      for s in 1:length(state_old)
        state_new[s, q, e] = state_new_q[s]
      end
    end
    
    @views _assemble_element!(assembler, Val{:mass}(), K_el, conns[:, e], e, block_id)
  end
  return nothing
end


# GPU implementation

KA.@kernel function _assemble_block_mass_kernel!(
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
  K_el = zeros(SMatrix{NxNDof, NxNDof, Float64, NxNDof * NxNDof})

  for q in 1:num_quadrature_points(ref_fe)
    interps = ref_fe.cell_interps.vals[q]
    state_old_q = _quadrature_level_state(state_old, q, E)
    K_q, state_new_q = mass(physics, interps, u_el, x_el, state_old_q, props_el, dt)
    K_el = K_el + K_q
    for s in 1:length(state_old)
      state_new[s, q, E] = state_new_q[s]
    end
  end

  block_size = values(assembler.pattern.block_sizes)[block_id]
  block_offset = values(assembler.pattern.block_offsets)[block_id]
  start_id = (block_id - 1) * block_size + 
             (E - 1) * block_offset + 1
  end_id = start_id + block_offset - 1
  ids = start_id:end_id
  for (i, id) in enumerate(ids)
    Atomix.@atomic assembler.mass_storage[id] += K_el.data[i]
  end
end

"""
$(TYPEDSIGNATURES)
Assembly method for a block labelled as block_id. This is a GPU agnostic implementation
using KernelAbstractions and Atomix for eliminating race conditions

TODO add state variables and physics properties
"""
function _assemble_block_mass!(
  assembler, physics, ref_fe, 
  U, X, state_old, state_new, props, dt,
  conns, block_id, backend::KA.Backend
)
  kernel! = _assemble_block_mass_kernel!(backend)
  kernel!(
    assembler, physics, ref_fe, 
    U, X, state_old, state_new, props, dt,
    conns, block_id, ndrange=size(conns, 2)
  )
  return nothing
end
