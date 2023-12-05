module FiniteElementContainers

# Meshes exports
export FileMesh
export Mesh
export connectivity
export coordinates
export element_type
export num_dimensions
export num_nodes

# Fie exports
export Field
export NodalField

# Domain exports
export Domain

abstract type FEMContainer end

include("Meshes.jl")

include("Fields.jl")

include("Domains.jl")

end # module