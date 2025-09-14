module FiniteElementContainersAMDGPUExt

using Adapt
using AMDGPU
using FiniteElementContainers
using KernelAbstractions

# need to double check if it's ROCArray
FiniteElementContainers.rocm(x) = Adapt.adapt_structure(ROCArray, x)

# this method need to have the assembler initialized first
# if the stored values in asm.pattern.cscnzval or zero
# AMDGPU will error out
function AMDGPU.rocSPARSE.ROCSparseMatrixCSC(asm::SparseMatrixAssembler)
  # Not sure if the line below is right on AMD
  @assert typeof(get_backend(asm)) <: ROCBackend "Assembler is not on a AMDGPU device"
  @assert length(asm.pattern.cscnzval) > 0 "Need to assemble the assembler once with SparseArrays.sparse!(assembler)"

  # below assertion isn't corect
  # @assert all(x -> x != zero(eltype(asm.pattern.cscnzval)), asm.pattern.cscnzval) "Need to assemble the assembler once with SparseArrays.sparse!(assembler)"
  #

  # n_dofs = FiniteElementContainers.num_unknowns(asm.dof)
  if FiniteElementContainers._is_condensed(asm.dof)
    n_dofs = length(asm.dof)
  else
    n_dofs = length(asm.dof.unknown_dofs)
  end

  return AMDGPU.rocSPARSE.ROCSparseMatrixCSC(
    asm.pattern.csccolptr,
    asm.pattern.cscrowval,
    asm.pattern.cscnzval,
    (n_dofs, n_dofs)
  )
end

function FiniteElementContainers._stiffness(asm::SparseMatrixAssembler, backend::ROCBackend)
  stiffness = AMDGPU.rocSPARSE.ROCSparseMatrixCSC(asm)
  return stiffness
end

end # module