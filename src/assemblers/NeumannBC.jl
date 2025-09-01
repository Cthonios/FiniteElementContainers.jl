# @inline function neumann_bc_energy(
#   bc::NeumannBCContainer,
#   u_el, x_el, t
# )
#   @assert false "In neumann bc residual"
# end

# @inline function neumann_bc_residual(
#   # bc::NeumannBCContainer,
#   interps, func,
#   u_el, x_el, t
# )
#   @assert false "In neumann bc residual"
# end

# # below method implicitly will not zero out arrays
"""
$(TYPEDSIGNATURES)
"""
function assemble_vector_neumann_bc!(
  assembler, Uu, p, ::Type{H1Field}
)
  storage = assembler.residual_storage
  # do not zero!
  t = current_time(p.times)
  # TODO should below 2 methods calls be assumed to have
  # been conducted previously?
  update_bcs!(p)
  update_field_unknowns!(p.h1_field, assembler.dof, Uu)

  for bc in values(p.neumann_bcs)
    backend = KA.get_backend(bc)
    _assemble_block_vector_neumann_bc!(
      storage, p.h1_field, p.h1_coords, t, 
      bc, backend
    )
  end
end

"""
$(TYPEDSIGNATURES)
"""
function _assemble_block_vector_neumann_bc!(
  field::F1, U::F2, X::F3, t::T,
  bc::N, ::KA.CPU
) where {
  F1 <: AbstractField,
  F2 <: AbstractField,
  F3 <: AbstractField,
  T  <: Number,
  N  <: NeumannBCContainer
}
  conns = bc.element_conns
  ref_fe = bc.ref_fe

  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(surface_element(ref_fe.element))
  NxNDof = NNPE * ND

  for e in axes(conns, 2)
    x_el = _element_level_fields(X, ref_fe, conns, e)
    R_el = zeros(SVector{NxNDof, eltype(field)})
    side = bc.bookkeeping.sides[e]
    for q in 1:num_quadrature_points(surface_element(ref_fe.element))
      interps = MappedSurfaceInterpolants(ref_fe, x_el, q, side)
      Nvec = interps.N_reduced
      JxW = interps.JxW

      # TODO Clean this up
      f_val = bc.vals[q, e]
      if length(f_val) == 1
        f_val = f_val[1]
      end

      R_el = R_el + JxW * Nvec * f_val
    end
    block_id = 1 # doesn't matter for this method
    el_id = bc.bookkeeping.elements[e]
    @views _assemble_element!(field, R_el, bc.surface_conns[:, e], el_id, block_id)
  end
end

"""
$(TYPEDSIGNATURES)
Kernel for residual block assembly of neumann bcs

TODO mark const fields
"""
# COV_EXCL_START
KA.@kernel function _assemble_block_vector_neumann_bc_kernel!(
  field::F1, U::F2, X::F3, t::T,
  bc::N
) where {
  F1 <: AbstractField,
  F2 <: AbstractField,
  F3 <: AbstractField,
  T  <: Number,
  N  <: NeumannBCContainer
}

  E = KA.@index(Global)
  conns = bc.element_conns
  ref_fe = bc.ref_fe

  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(surface_element(ref_fe.element))
  NxNDof = NNPE * ND

  x_el = _element_level_fields(X, ref_fe, conns, E)
  R_el = zeros(SVector{NxNDof, eltype(field)})
  side = bc.bookkeeping.sides[E]

  for q in 1:num_quadrature_points(surface_element(ref_fe.element))
    interps = MappedSurfaceInterpolants(ref_fe, x_el, q, side)
    Nvec = interps.N_reduced
    JxW = interps.JxW

    # TODO Clean this up
    f_val = bc.vals[q, E]
    if length(f_val) == 1
      f_val = f_val[1]
    end

    R_el = R_el + JxW * Nvec * f_val
  end

  # now assemble atomically
  n_dofs = size(field, 1)
  for i in 1:size(bc.surface_conns, 1)
    for d in 1:n_dofs
      global_id = n_dofs * (bc.surface_conns[i, E] - 1) + d
      local_id = n_dofs * (i - 1) + d
      Atomix.@atomic field.data[global_id] += R_el[local_id]
    end
  end
end
# COV_EXCL_STOP

"""
$(TYPEDSIGNATURES)
"""
function _assemble_block_vector_neumann_bc!(
  field::F1, U::F2, X::F3, t::T,
  bc::N, backend::KA.Backend
) where {
  F1 <: AbstractField,
  F2 <: AbstractField,
  F3 <: AbstractField,
  T  <: Number,
  N  <: NeumannBCContainer
}
  kernel! = _assemble_block_vector_neumann_bc_kernel!(backend)
  kernel!(
    field, U, X, t, bc, ndrange=size(bc.vals, 2)
  )
  return nothing
end
