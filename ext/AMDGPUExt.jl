module AMDGPUExt

import Adapt
import AMDGPU
import FiniteElementContainers

# public API
function FiniteElementContainers.rocm(x)
  return Adapt.adapt_structure(AMDGPU.ROCArray, x)
end

# private API
function FiniteElementContainers._coo_matrix_constructor(::AMDGPU.ROCBackend)
  return AMDGPU.rocSPARSE.ROCSparseMatrixCOO
end

function FiniteElementContainers._csc_matrix_constructor(::AMDGPU.ROCBackend)
  return AMDGPU.rocSPARSE.ROCSparseMatrixCSC
end

function FiniteElementContainers._csr_matrix_constructor(::AMDGPU.ROCBackend)
  return AMDGPU.rocSPARSE.ROCSparseMatrixCSR
end

end # module
