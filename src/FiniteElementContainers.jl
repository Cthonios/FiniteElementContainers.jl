module FiniteElementContainers

# type exports
export DofManager,
       EssentialBC,
       FunctionSpace,
       Mesh

# method exports
export create_fields,
       create_unknowns,
       dof_connectivity,
       element_connectivity,
       update_bcs!,
       update_fields!

# dependencies
using Exodus,
      Graphs,
      KernelAbstractions,
      LinearAlgebra,
      ReferenceFiniteElements,
      SparseArrays,
      StaticArrays,
      StructArrays

# include files
include("Meshes.jl")
include("EssentialBCs.jl")
include("FunctionSpaces.jl")
include("DofManagers.jl")

end # module
