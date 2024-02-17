module FiniteElementContainersExodusExt

using DocStringExtensions
using Exodus
using FiniteElementContainers

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
function FiniteElementContainers.FileMesh(type::Type{<:ExodusDatabase}, file_name::String)
  exo = type(file_name, "r")
  return FileMesh{typeof(exo)}(file_name, exo)
end

function FiniteElementContainers.num_dimensions(
  mesh::FileMesh{<:ExodusDatabase}
)::Int32
  return Exodus.num_dimensions(mesh.mesh_obj.init)
end

function FiniteElementContainers.num_nodes(
  mesh::FileMesh{<:ExodusDatabase}
)::Int32
  return Exodus.num_nodes(mesh.mesh_obj.init)
end

"""
$(TYPEDSIGNATURES)
"""
function FiniteElementContainers.element_block_ids(mesh::FileMesh{<:ExodusDatabase})
  return Exodus.read_ids(mesh.mesh_obj, Block)
end

"""
$(TYPEDSIGNATURES)
"""
function FiniteElementContainers.nodeset_ids(mesh::FileMesh{<:ExodusDatabase})
  return Exodus.read_ids(mesh.mesh_obj, NodeSet)
end

"""
$(TYPEDSIGNATURES)
"""
function FiniteElementContainers.sideset_ids(mesh::FileMesh{<:ExodusDatabase})
  return Exodus.read_ids(mesh.mesh_obj, SideSet)
end

"""
$(TYPEDSIGNATURES)
"""
function FiniteElementContainers.coordinates(mesh::FileMesh{ExodusDatabase{M, I, B, F}})::Matrix{F} where {M, I, B, F} 
  coords = Exodus.read_coordinates(mesh.mesh_obj)
  return coords
end

function FiniteElementContainers.element_connectivity(
  mesh::FileMesh{<:ExodusDatabase},
  id::Integer
) 

  block = read_block(mesh.mesh_obj, id)
  return block.conn
end

function FiniteElementContainers.element_type(
  mesh::FileMesh{<:ExodusDatabase},
  id::Integer
) 
  block = read_block(mesh.mesh_obj, id)
  return block.elem_type
end

function FiniteElementContainers.copy_mesh(file_1::String, file_2::String)
  Exodus.copy_mesh(file_1, file_2)
end

function FiniteElementContainers.nodeset(
  mesh::FileMesh{<:ExodusDatabase},
  id::Integer
) 
  nset = read_set(mesh.mesh_obj, NodeSet, id)
  return nset.nodes
end

function FiniteElementContainers.sideset(
  mesh::FileMesh{<:ExodusDatabase},
  id::Integer
) 
  sset = read_set(mesh.mesh_obj, SideSet, id)
  return sset.elements, sset.sides
end

end # module
