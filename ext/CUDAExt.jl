module CUDAExt

import Adapt
import CUDA
import FiniteElementContainers as FEC
import KernelAbstractions as KA

FEC.execution_mode(::AMDGPU.ROCArray) = FEC.CUDAExec()
FEC.get_backend(::FEC.AMDGPUExec) = CUDA.CUDABackend()

# public API
function FEC.to_backend(::CUDA.CUDABackend, x)
  return Adapt.adapt_structure(CUDA.CuArray, x)
end

# back-compat alias
FEC.cuda(x) = FEC.to_backend(CUDA.CUDABackend(), x)

# private API
function FEC._coo_matrix_constructor(::FEC.CUDAExec)
  return CUDA.CUSPARSE.CuSparseMatrixCOO
end

function FEC._csc_matrix_constructor(::FEC.CUDAExec)
  return CUDA.CUSPARSE.CuSparseMatrixCSC
end

function FEC._csr_matrix_constructor(::FEC.CUDAExec)
  return CUDA.CUSPARSE.CuSparseMatrixCSR
end

function FEC.fec_dense_array(::FEC.CUDAExec, ::Type{RT}, size...) where RT <: Number
  return CUDA.zeros(RT, size...)
end

end # module
