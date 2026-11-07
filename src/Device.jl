# TODO we'll need to eventually specialize
# some of these
abstract type AbstractExec end
abstract type GPUExec <: AbstractExec end

struct AMDGPUExec <: GPUExec
end

struct CPUExec <: AbstractExec
end

struct CUDAExec <: GPUExec
end

function execution_mode(::AbstractArray)
    return CPUExec()
end

# function execution_mode(::GPUArrays.AbstractGPUSparseMatrix)
#   return 
# end

get_backend(a::T) where T = KI.get_backend(a)
function get_host end

"""
$(TYPEDSIGNATURES)
Move `x` onto the given KernelAbstractions backend.  CPU is identity —
CPU-built data already lives on the CPU backend.  GPU backends (CUDABackend,
ROCBackend) are provided by the CUDA / AMDGPU package extensions.
"""
to_backend(b::KI.Backend, x) = error(
  "to_backend is not implemented for backend $(typeof(b)); load the " *
  "corresponding GPU package (CUDA.jl or AMDGPU.jl) so its extension activates."
)

# to_backend(::KA.CPU, x) = x

# """
# $(TYPEDSIGNATURES)
# Move `x` onto the host.
# """
# to_host(x) = to_backend(KA.CPU(), x)

#########################################
# hooks for extensions
#########################################
# device stuff
function cpu end
# TODO need to further specialize for staticarrays, etc.
cpu(x) = adapt(Array, x)
function cuda end
function rocm end
