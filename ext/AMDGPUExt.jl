module AMDGPUExt

import Adapt
import AMDGPU
import FiniteElementContainers as FEC

FEC.execution_mode(::AMDGPU.ROCArray) = FEC.AMDGPUExec()
FEC.execution_mode(::AMDGPU.rocSPARSE.AbstractROCSparseMatrix) = FEC.AMDGPUExec()
FEC.get_backend(::FEC.AMDGPUExec) = AMDGPU.ROCBackend()

# public API
function FEC.to_backend(::AMDGPU.ROCBackend, x)
  return Adapt.adapt_structure(AMDGPU.ROCArray, x)
end

# back-compat alias
FEC.rocm(x) = FEC.to_backend(AMDGPU.ROCBackend(), x)

# private API
function FEC._coo_matrix_constructor(::FEC.AMDGPUExec)
  return AMDGPU.rocSPARSE.ROCSparseMatrixCOO
end

function FEC._csc_matrix_constructor(::FEC.AMDGPUExec)
  return AMDGPU.rocSPARSE.ROCSparseMatrixCSC
end

function FEC._csr_matrix_constructor(::FEC.AMDGPUExec)
  return AMDGPU.rocSPARSE.ROCSparseMatrixCSR
end

function FEC.fec_dense_array(::FEC.AMDGPUExec, ::Type{RT}, size...) where RT <: Number
  return AMDGPU.zeros(RT, size...)
end

end # module
