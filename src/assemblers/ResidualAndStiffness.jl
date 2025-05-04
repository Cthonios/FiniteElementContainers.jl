# Top level method
function assemble!(assembler, ::Type{H1Field}, p, val_sym::Val{:residual_and_stiffness})
  fspace = assembler.dof.H1_vars[1].fspace
  _zero_storage(assembler, val_sym)
  for (b, (conns, block_physics, state_old, state_new, props)) in enumerate(zip(
    values(fspace.elem_conns), 
    values(p.physics),
    values(p.state_old), values(p.state_new),
    values(p.properties)
  ))
    ref_fe = values(fspace.ref_fes)[b]
    backend = _check_backends(assembler, p.h1_field, p.h1_coords, state_old, state_new, conns)
    _assemble_block_residual_and_stiffness!(
      assembler, block_physics, ref_fe, 
      p.h1_field, p.h1_coords, state_old, state_new, props,
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
TODO remove Float64 typing below for eventual unitful use
"""
function _assemble_block_residual_and_stiffness!(
  assembler, physics, ref_fe, 
  U, X, state_old, state_new, props,
  conns, block_id, ::KA.CPU
)
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
  for e in axes(conns, 2)
    x_el = _element_level_fields(X, ref_fe, conns, e)
    u_el = _element_level_fields(U, ref_fe, conns, e)
    R_el = zeros(SVector{NxNDof, eltype(assembler.residual_storage)})
    K_el = zeros(SMatrix{NxNDof, NxNDof, eltype(assembler.stiffness_storage), NxNDof * NxNDof})

    for q in 1:num_quadrature_points(ref_fe)
      interps = MappedInterpolants(ref_fe.cell_interps.vals[q], x_el)
      R_q = residual(physics, interps, u_el)
      K_q = stiffness(physics, interps, u_el)
      R_el = R_el + R_q
      K_el = K_el + K_q
    end
    
    @views _assemble_element!(assembler.residual_storage, R_el, conns[:, e], e, block_id)
    @views _assemble_element!(assembler, Val{:stiffness}(), K_el, conns[:, e], e, block_id)
  end
  return nothing
end

# TODO implement GPU version