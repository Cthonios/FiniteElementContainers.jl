# Assembly of body force contributions to the residual vector.
#
# Computes -∫ Nᵢ · b dΩ for each element block where body forces are defined.
# Does NOT zero the residual storage — adds to the existing assembled residual
# (same convention as NeumannBC assembly).

"""
$(TYPEDSIGNATURES)
Assemble body force contributions into the residual vector.
"""
function assemble_vector_source!(assembler, Uu, p)
  assemble_vector_source!(
    assembler.residual_storage,
    assembler.vector_pattern, assembler.dof,
    Uu, p
  )
  return nothing
end

function assemble_vector_source!(storage, pattern, dof, Uu, p)
  isempty(p.sources.source_caches) && return nothing

  _update_for_assembly!(p, dof, Uu)
  fspace = function_space(dof)
  U = p.field
  X = coordinates(p)
  sources = p.sources
  conns = fspace.elem_conns
  for (block_id, source) in zip(
    sources.source_block_ids, sources.source_caches
  )
    # ref_fe = getfield(fspace.ref_fes, block_name)
    ref_fe = fspace.ref_fes[block_id]
    vals   = source.vals

    _assemble_block_vector_source!(
      storage,
      conns.data, conns.offsets[block_id],
      ref_fe, X, U, vals,
    )
  end
  return nothing
end

function _assemble_block_vector_source!(
  field::AbstractField, conns, coffset, ref_fe, X, U, vals
)
  fec_foraxes(vals, 2) do e
    conn = connectivity(ref_fe, conns, e, coffset)
    X_el = _element_level_fields_flat(X, ref_fe, conn)
  
    for q in 1:num_cell_quadrature_points(ref_fe)
      interps = _cell_interpolants(ref_fe, q)
      interps = map_interpolants(interps, X_el)
      N = interps.N
      JxW  = interps.JxW
      b_val = vals[q, e]
  
      form = GeneralFormulation{num_fields(X), num_fields(field)}()
      scatter_with_values!(field, form, e, conn, N, -JxW * b_val)
    end
  end
end
