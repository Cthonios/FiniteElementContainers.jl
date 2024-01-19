"""
$(TYPEDEF)
"""
abstract type AbstractMesh <: FEMContainer end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function coordinates end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function copy_mesh end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function element_block_ids end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function element_connectivity end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function element_type end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function nodeset end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function nodeset_ids end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function sideset end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function sideset_ids end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function num_dimensions end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function num_nodes end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""

"""
$(TYPEDSIGNATURES)
Returns file name for an mesh type
"""
file_name(mesh::AbstractMesh) = mesh.file_name

# TODO this one is making JET not happy
"""
$(TYPEDEF)
$(TYPEDFIELDS)
Mesh type that has a handle to an open mesh file object.
This type's methods are "overridden" in extensions.

See FiniteElementContainersExodusExt for an example.
"""
struct FileMesh{MeshObj} <: AbstractMesh
  file_name::String
  mesh_obj::MeshObj
end
# function num_dimensions(::FileMesh{ND, NN, M})::Int32 where {ND, NN, M}
# num_nodes(::FileMesh{ND, NN, M}) where {ND, NN, M} = NN

# struct DummyMeshObj
# end

# dummy function to make JET happy
# file_mesh(::Type{DummyMeshObj}, ::String) = nothing
