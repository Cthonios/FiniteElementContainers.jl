module CUDAExt

import Adapt
import CUDA
import FiniteElementContainers as FEC
import KernelAbstractions as KA

FEC.is_cpu(::CUDA.CUDABackend) = false
FEC.is_gpu(::CUDA.CUDABackend) = true

# public API
function FEC.to_backend(::CUDA.CUDABackend, x)
  return Adapt.adapt_structure(CUDA.CuArray, x)
end

# back-compat alias
FEC.cuda(x) = FEC.to_backend(CUDA.CUDABackend(), x)

# private API
function FEC._coo_matrix_constructor(::CUDA.CUDABackend)
  return CUDA.CUSPARSE.CuSparseMatrixCOO
end

function FEC._csc_matrix_constructor(::CUDA.CUDABackend)
  return CUDA.CUSPARSE.CuSparseMatrixCSC
end

function FEC._csr_matrix_constructor(::CUDA.CUDABackend)
  return CUDA.CUSPARSE.CuSparseMatrixCSR
end

function FEC.fec_dense_array(::CUDA.CUDABackend, ::Type{RT}, size...) where RT <: Number
  return CUDA.zeros(RT, size...)
end

end # module
