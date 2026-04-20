import KernelAbstractions as KA
using TestItems

function _check_functional_backend(name)
    if isdefined(Main, name)
        return eval(name).functional()
    else
        return false
    end
end
  
function _backend_to_array_type(backend::Function)
    if isa(backend, typeof(cpu))
        return Array
    elseif isa(backend, typeof(cuda))
        return CUDA.CuArray
    elseif isa(backend, typeof(rocm))
        return AMDGPU.ROCArray
    else
        @assert false "Unsupported backend $backend"
    end
end

function _get_backends()
    if "--ignore-cpu" in ARGS
        backends = Function[]
    else 
        backends = Function[cpu]
    end
    if _check_functional_backend(:AMDGPU)
        push!(backends, rocm)
    end
    if _check_functional_backend(:CUDA)
        push!(backends, cuda)
    end
    return backends
end

@testsnippet TestHelpers begin
    backends = _get_backends()
end
