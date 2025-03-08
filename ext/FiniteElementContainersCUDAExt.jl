module FiniteElementContainersCUDAExt

using CUDA
using FiniteElementContainers
using KernelAbstractions

function CUDA.CUSPARSE.CuSparseMatrixCSC(asm::SparseMatrixAssembler)
  @assert typeof(get_backend(asm)) <: CUDABackend
  n_dofs = FiniteElementContainers.num_unknowns(asm.dof)
  return CUDA.CUSPARSE.CuSparseMatrixCSC(
    asm.pattern.csccolptr,
    asm.pattern.cscrowval,
    asm.pattern.cscnzval,
    (n_dofs, n_dofs)
  )
end

# this one isn't quite right
# function CUDA.CUSPARSE.CuSparseMatrixCSR(asm::SparseMatrixAssembler)
#   n_dofs = FiniteElementContainers.num_unknowns(asm.dof)
#   return CUDA.CUSPARSE.CuSparseMatrixCSR(
#     asm.pattern.csrrowptr,
#     asm.pattern.csrcolval,
#     asm.pattern.csrnzval,
#     (n_dofs, n_dofs)
#   )
# end

end # module