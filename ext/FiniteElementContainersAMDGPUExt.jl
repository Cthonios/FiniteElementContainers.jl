module FiniteElementContainersAMDGPUExt

using Adapt
using AMDGPU
using FiniteElementContainers
using KernelAbstractions

FiniteElementContainers.gpu(x) = adapt_structure(ROCArray, x)

function AMDGPU.rocSPARSE.ROCSparseMatrixCSC(asm::SparseMatrixAssembler)
  # TODO Not sure what the AMD Backend is called in KernelAbstractions
  # I couldn't quite figure it out. This assert statement below though
  # would be good for error checking and device consistency.
  # @assert typeof(get_backend(asm)) <: CUDABackend "Assembler is not on a CUDA device"
  @assert length(asm.pattern.cscnzval) > 0 "Need to assemble the assembler once with SparseArrays.sparse!(assembler)"
  @assert all(x -> x != zero(eltype(asm.pattern.cscnzval)), asm.pattern.cscnzval) "Need to assemble the assembler once with SparseArrays.sparse!(assembler)"
  n_dofs = FiniteElementContainers.num_unknowns(asm.dof)
  return AMDGPU.rocSPARSE.ROCSparseMatrixCSC(
    asm.pattern.csccolptr,
    asm.pattern.cscrowval,
    asm.pattern.cscnzval,
    (n_dofs, n_dofs)
  )
end

end # module
