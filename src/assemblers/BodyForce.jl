# Assembly of body force contributions to the residual vector.
#
# Computes -∫ Nᵢ · b dΩ for each element block where body forces are defined.
# Does NOT zero the residual storage — adds to the existing assembled residual
# (same convention as NeumannBC assembly).

"""
$(TYPEDSIGNATURES)
Assemble body force contributions into the residual vector.
"""
function assemble_vector_body_force!(assembler, Uu, p)
  isempty(p.body_forces.bc_caches) && return nothing

  _update_for_assembly!(p, assembler.dof, Uu)
  U = p.field
  X = coordinates(p)

  backend = KA.get_backend(assembler)

  for bc in values(p.body_forces.bc_caches)
    ref_fe = bc.ref_fe
    conns  = bc.element_conns.data
    vals   = bc.vals
    nelem  = length(bc)

    _assemble_block_vector_body_force!(
      backend,
      assembler.residual_storage,
      conns, nelem,
      ref_fe, X, U, vals,
    )
  end
  return nothing
end

# CPU implementation
function _assemble_block_vector_body_force!(
  ::KA.CPU,
  field,
  conns, nelem::Int,
  ref_fe::ReferenceFE,
  X::AbstractField, U::AbstractField,
  vals,
)
  for e in 1:nelem
    conn = connectivity(ref_fe, conns, e, 1)
    x_el = _element_level_fields(X, ref_fe, conn)
    R_el = _element_scratch(AssembledVector(), ref_fe, U)

    for q in 1:num_cell_quadrature_points(ref_fe)
      interps = MappedH1OrL2Interpolants(ref_fe, x_el, q)
      Nvec = interps.N     # array of scalar shape function values [N₁, N₂, ..., Nₙ]
      JxW  = interps.JxW
      b_val = vals[q, e]   # SVector{ND} body force density at this QP

      # Accumulate: R_el[ND*(i-1)+α] -= JxW * Nᵢ * bα
      # Negated so that positive b = force in positive direction (physics convention).
      # The residual is R = F_int - F_ext; body forces are F_ext, hence subtracted.
      R_el = R_el - JxW * reduce(vcat, ntuple(i -> Nvec[i] * b_val, length(Nvec)))
    end

    @views _assemble_element!(field, R_el, conn, e, 0, 0)
  end
  return nothing
end

# GPU implementation
# COV_EXCL_START
KA.@kernel function _assemble_block_vector_body_force_kernel!(
  field,
  conns,
  ref_fe::ReferenceFE,
  X::AbstractField, U::AbstractField,
  vals,
)
  E = KA.@index(Global)
  conn = connectivity(ref_fe, conns, E, 1)
  x_el = _element_level_fields(X, ref_fe, conn)
  R_el = _element_scratch(AssembledVector(), ref_fe, U)

  for q in 1:num_cell_quadrature_points(ref_fe)
    interps = MappedH1OrL2Interpolants(ref_fe, x_el, q)
    Nvec = interps.N
    JxW  = interps.JxW
    b_val = vals[q, E]

    R_el = R_el - JxW * reduce(vcat, ntuple(i -> Nvec[i] * b_val, length(Nvec)))
  end

  @views _assemble_element!(field, R_el, conn, E, 0, 0)
end
# COV_EXCL_STOP

function _assemble_block_vector_body_force!(
  backend::KA.Backend,
  field,
  conns, nelem::Int,
  ref_fe, X, U, vals,
)
  kernel! = _assemble_block_vector_body_force_kernel!(backend)
  kernel!(field, conns, ref_fe, X, U, vals, ndrange=nelem)
  return nothing
end
