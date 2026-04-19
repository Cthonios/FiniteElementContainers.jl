using TestItems

function _check_functional_backend(name)
    if isdefined(Main, name)
        return eval(name).functional()
    else
        return false
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
