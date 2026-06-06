module CUDAExt

import Adapt
import CUDA
import FiniteElementContainers

# public API
function FiniteElementContainers.to_backend(::CUDA.CUDABackend, x)
  return Adapt.adapt_structure(CUDA.CuArray, x)
end

# back-compat alias
FiniteElementContainers.cuda(x) =
  FiniteElementContainers.to_backend(CUDA.CUDABackend(), x)

# private API
function FiniteElementContainers._coo_matrix_constructor(::CUDA.CUDABackend)
  return CUDA.CUSPARSE.CuSparseMatrixCOO
end

function FiniteElementContainers._csc_matrix_constructor(::CUDA.CUDABackend)
  return CUDA.CUSPARSE.CuSparseMatrixCSC
end

function FiniteElementContainers._csr_matrix_constructor(::CUDA.CUDABackend)
  return CUDA.CUSPARSE.CuSparseMatrixCSR
end

function FiniteElementContainers.fec_dense_array(::CUDA.CUDABackend, size...)
  return CUDA.zeros(size...)
end

function FiniteElementContainers.fec_dense_array(::CUDA.CUDABackend, ::Type{RT}, size...) where RT <: Number
  return CUDA.zeros(RT, size...)
end

end # module
