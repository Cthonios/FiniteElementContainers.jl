function assemble_mass!(
  assembler, func::F, Uu, p
) where F <: Function
  _check_matrix_assembly_supported(assembler, "assemble_mass!")
  assemble_matrix!(
    assembler.mass_storage, assembler.matrix_pattern, assembler.dof,
    func, Uu, p;
    use_inplace_methods = _use_inplace_methods(assembler)
  )
end

function assemble_stiffness!(
  assembler::SparseMatrixAssembler, func::F, Uu, p
) where F <: Function
  _check_matrix_assembly_supported(assembler, "assemble_stiffness!")
  assemble_matrix!(
    assembler.stiffness_storage, assembler.matrix_pattern, assembler.dof,
    func, Uu, p;
    use_inplace_methods = _use_inplace_methods(assembler)
  )
end

@inline function _check_matrix_assembly_supported(asm, fname::AbstractString)
  if asm isa SparseMatrixAssembler && _is_matrix_free(asm)
    error("$fname called on a matrix-free SparseMatrixAssembler.  " *
          "Re-create the assembler with matrix_free=false to enable matrix assembly.")
  end
end

"""
$(TYPEDSIGNATURES)
Note this is hard coded to storing the assembled sparse matrix in 
the stiffness_storage field of assembler.
"""
function assemble_matrix!(
  storage, pattern, dof, func::F, Uu, p;
  use_inplace_methods::Bool = false
) where F <: Function
  fill!(storage, zero(eltype(storage)))
  fspace = function_space(dof)
  X = coordinates(p)
  t = current_time(p)
  dt = time_step(p)
  U = p.field
  U_old = p.field_old
  _update_for_assembly!(p, dof, Uu)
  return_type = AssembledMatrix()
  conns = fspace.elem_conns
  foreach_block(fspace, p) do physics, ref_fe, b
    if use_inplace_methods
      _assemble_block!(
        block_view(storage, pattern, b),
        func,
        b,
        physics,
        t, dt,
        p.properties, p.state_old, p.state_new,
        conns,
        ref_fe, X, U, U_old
      )
    else
      _assemble_block!(
        block_view(storage, pattern, b),
        conns,
        func,
        b,
        physics, ref_fe,
        X, t, dt,
        U, U_old, 
        p.state_old, p.state_new, p.properties,
        return_type
      )
    end
  end

  return nothing
end
