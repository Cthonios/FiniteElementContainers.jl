module KernelAbstractionsExt

import FiniteElementContainers as FEC
import KernelAbstractions as KA

FEC.get_host() = KA.CPU()
FEC.to_backend(::KA.CPU, x) = x

end # module
