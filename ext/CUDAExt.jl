module CUDAExt

import Adapt
import CUDA
import FiniteElementContainers

# public API
function FiniteElementContainers.cuda(x)
  return Adapt.adapt_structure(CUDA.CuArray, x)
end

# private API
function FiniteElementContainers._coo_matrix_constructor(::CUDA.CUDABackend)
  return CUDA.CUSPARSE.CuSparseMatrixCOO
end

function FiniteElementContainers._csc_matrix_constructor(::CUDA.CUDABackend)
  return CUDA.CUSPARSE.CuSparseMatrixCSC
end

end # module
