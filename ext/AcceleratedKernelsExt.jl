module AcceleratedKernelsExt

import AcceleratedKernels as AK
import FiniteElementContainers as FEC
import KernelAbstractions as KA

# General GPU implementation leveraging
# the Accelerated kernels implementation
function FEC._fec_forindices(
    backend::KA.Backend, f, indices,
    block_size::Int, max_tasks::Int, min_elems::Int
)
    AK._forindices_gpu(f, indices, backend; block_size = block_size)
end

end # module
