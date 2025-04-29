# """
# $(TYPEDSIGNATURES)
# Assembly method for a block labelled as block_id. This is a CPU implementation
# with no threading.

# TODO add state variables and physics properties
# TODO remove Float64 typing below for eventual unitful use
# """
# function _assemble_block!(assembler, physics, ::Val{:residual}, ref_fe, U, X, conns, block_id, ::KA.CPU)
#   ND = size(U, 1)
#   NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
#   NxNDof = NNPE * ND
#   for e in axes(conns, 2)
#     x_el = _element_level_coordinates(X, ref_fe, conns, e)
#     u_el = _element_level_fields(U, ref_fe, conns, e)
#     R_el = zeros(SVector{NxNDof, Float64})

#     for q in 1:num_quadrature_points(ref_fe)
#       interps = MappedInterpolants(ref_fe.cell_interps.vals[q], x_el)
#       R_q = residual(physics, interps, u_el)
#       R_el = R_el + R_q
#     end
    
#     @views _assemble_element!(assembler.residual_storage, R_el, conns[:, e], e, block_id)
#   end
# end

"""
$(TYPEDSIGNATURES)
Assembly method for a block labelled as block_id. This is a CPU implementation
with no threading.

TODO add state variables and physics properties
TODO remove Float64 typing below for eventual unitful use
"""
function _assemble_block_residual!(assembler, physics, ref_fe, U, X, conns, block_id, ::KA.CPU)
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
  for e in axes(conns, 2)
    x_el = _element_level_fields(X, ref_fe, conns, e)
    u_el = _element_level_fields(U, ref_fe, conns, e)
    R_el = zeros(SVector{NxNDof, eltype(assembler.residual_storage)})

    for q in 1:num_quadrature_points(ref_fe)
      interps = MappedInterpolants(ref_fe.cell_interps.vals[q], x_el)
      R_q = residual(physics, interps, u_el)
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
