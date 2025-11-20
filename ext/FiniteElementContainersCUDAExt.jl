module FiniteElementContainersCUDAExt

using Adapt
using CUDA
using FiniteElementContainers
using KernelAbstractions

FiniteElementContainers.cuda(x) = adapt(CuArray, x)

# this method need to have the assembler initialized first
# if the stored values in asm.pattern.cscnzval or zero
# CUDA will error out
function CUDA.CUSPARSE.CuSparseMatrixCSC(asm::SparseMatrixAssembler)
  @assert typeof(get_backend(asm)) <: CUDABackend "Assembler is not on a CUDA device"
  # TODO these assert statements are failing for multi dof problems yet
  # they are doing the right thing
  # @assert length(asm.pattern.cscnzval) > 0 "Need to assemble the assembler once with SparseArrays.sparse!(assembler)"
  # @assert all(x -> x != zero(eltype(asm.pattern.cscnzval)), asm.pattern.cscnzval) "Need to assemble the assembler once with SparseArrays.sparse!(assembler)"
  if FiniteElementContainers._is_condensed(asm.dof)
    n_dofs = length(asm.dof)
  else
    n_dofs = length(asm.dof.unknown_dofs)
  end

  return CUDA.CUSPARSE.CuSparseMatrixCSC(
    asm.matrix_pattern.csccolptr,
    asm.matrix_pattern.cscrowval,
    asm.matrix_pattern.cscnzval,
    (n_dofs, n_dofs)
  )
end

function FiniteElementContainers._stiffness(asm::SparseMatrixAssembler, ::CUDABackend)
  return CUDA.CUSPARSE.CuSparseMatrixCSC(asm)
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