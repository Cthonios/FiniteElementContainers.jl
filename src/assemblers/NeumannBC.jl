function assemble_vector_neumann_bc!(
    assembler, Uu, p, ::Type{H1Field}
)
  # do we want to zero this out?
  # should all assemblers not handle zeroing?
  # and leave it to the solvers?

  fspace = function_space(assembler, H1Field)
  t = current_time(p.times)
  update_bcs!(p)
  update_field_unknowns!(p.h1_field, assembler.dof, Uu)
  
  # loop over blocks then bcs
  # we'll specialize the method on block type
  # and loop over bcs in the method
  for (b, conns) in enumerate(values(fspace.elem_conns))
    ref_fe = values(fspace.ref_fes)[b]
    # TODO check backends
    backend = KA.get_backend(assembler)
    _assemble_block_vector_neumann_bc!(
      assembler.residual_storage, ref_fe, 
      p.h1_field, p.h1_coords, t,
      conns, b,
      p.neumann_bcs,
      backend
    )
  end 
end

function _assemble_block_vector_neumann_bc!(
  field::F1, ref_fe::R, U::F2, X::F3, t::T, 
  conns::C, block_id::Int, bcs::N, ::KA.CPU
) where {
  C  <: Connectivity,
  F1 <: AbstractField,
  F2 <: AbstractField,
  F3 <: AbstractField,
  N  <: NamedTuple,
  R  <: ReferenceFE,
  T  <: Number
}

  for bc in values(bcs)
    for (n, e) in bc.bookkeeping.elements

    end
  end
end

function _assemble_block_vector_neumann_bc!(
  field::F1, ref_fe::R, U::F2, X::F3, t::T, 
  conns::C, block_id::Int, bcs::N, ::KA.Backend
) where {
  C  <: Connectivity,
  F1 <: AbstractField,
  F2 <: AbstractField,
  F3 <: AbstractField,
  N  <: NamedTuple,
  R  <: ReferenceFE,
  T  <: Number
}
  @warn "Neumann BCs not functional on GPU yet"
end