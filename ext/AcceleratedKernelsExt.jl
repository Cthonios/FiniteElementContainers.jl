module AcceleratedKernelsExt

import AcceleratedKernels as AK
import FiniteElementContainers as FEC
import KernelAbstractions as KA

function FEC._forindices_gpu(f, indices, backend, block_size)
    AK._forindices_gpu(f, indices, backend; block_size)
end

function FEC.use_gpu(backend::KA.Backend, prefer_threads::Bool)
    return AK.use_gpu_algorithm(backend, prefer_threads)
end

end # module
