module FiniteElementContainers

# type exports
export Connectivity,
       DofManager,
       DynamicAssembler,
       EssentialBC,
       FunctionSpace,
       Mesh,
       StaticAssembler

# method exports
export assemble!,
       create_fields,
       create_unknowns,
       dof_connectivity,
       element_connectivity,
       update_bcs!,
       update_fields!

# dependencies
using Exodus,
      KernelAbstractions,
      LinearAlgebra,
      ReferenceFiniteElements,
      SparseArrays,
      StaticArrays,
      StructArrays

# include files
include("Meshes.jl")
include("EssentialBCs.jl")
include("Connectivities.jl")
include("FunctionSpaces.jl")
# include("Connectivities.jl")
include("DofManagers.jl")
include("Assemblers.jl")

end # module
