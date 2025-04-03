"""
$(TYPEDSIGNATURES)
Kernel for residual block assembly
"""
KA.@kernel function _assemble_block_residual_kernel!(assembler, physics, ref_fe, U, X, conns, block_id)
  E = KA.@index(Global)

  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND

  x_el = _element_level_coordinates(X, ref_fe, conns, E)
  u_el = _element_level_fields(U, ref_fe, conns, E)
  R_el = zeros(SVector{NxNDof, Float64})

  for q in 1:num_quadrature_points(ref_fe)
    interps = MappedInterpolants(ref_fe.cell_interps.vals[q], x_el)
    R_q = residual(physics, interps, u_el)
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
function _assemble_block!(assembler, physics, ::Val{:residual}, ref_fe, U, X, conns, block_id, backend::KA.Backend)
  kernel! = _assemble_block_residual_kernel!(backend)
  kernel!(assembler, physics, ref_fe, U, X, conns, block_id, ndrange=size(conns, 2))
  return nothing
end

KA.@kernel function _assemble_block_stiffness_kernel!(assembler, physics, ref_fe, U, X, conns, block_id)
  E = KA.@index(Global)

  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND

  x_el = _element_level_coordinates(X, ref_fe, conns, E)
  u_el = _element_level_fields(U, ref_fe, conns, E)
  K_el = zeros(SMatrix{NxNDof, NxNDof, Float64, NxNDof * NxNDof})

  for q in 1:num_quadrature_points(ref_fe)
    interps = MappedInterpolants(ref_fe.cell_interps.vals[q], x_el)
    K_q = stiffness(physics, interps, u_el)
    K_el = K_el + K_q
  end

  block_size = values(assembler.pattern.block_sizes)[block_id]
  block_offset = values(assembler.pattern.block_offsets)[block_id]
  start_id = (block_id - 1) * block_size + 
             (E - 1) * block_offset + 1
  end_id = start_id + block_offset - 1
  ids = start_id:end_id
  for (i, id) in enumerate(ids)
    Atomix.@atomic assembler.stiffness_storage[id] += K_el.data[i]
  end
end

"""
$(TYPEDSIGNATURES)
Assembly method for a block labelled as block_id. This is a GPU agnostic implementation
using KernelAbstractions and Atomix for eliminating race conditions

TODO add state variables and physics properties
"""
function _assemble_block!(assembler, physics, ::Val{:stiffness}, ref_fe, U, X, conns, block_id, backend::KA.Backend)
  kernel! = _assemble_block_stiffness_kernel!(backend)
  kernel!(assembler, physics, ref_fe, U, X, conns, block_id, ndrange=size(conns, 2))
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
