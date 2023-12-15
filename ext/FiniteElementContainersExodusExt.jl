module FiniteElementContainersExodusExt

using Exodus
using FiniteElementContainers

function FiniteElementContainers.FileMesh(::Type{ExodusDatabase}, file_name::String)
  exo = ExodusDatabase(file_name, "r")
  return FileMesh{typeof(exo)}(file_name, exo)
end

FiniteElementContainers.num_dimensions(
  mesh::FileMesh{<:ExodusDatabase}
)  = mesh.mesh_obj.init.num_dim

FiniteElementContainers.num_nodes(
  mesh::FileMesh{<:ExodusDatabase}
) = mesh.mesh_obj.init.num_nodes

function FiniteElementContainers.element_block_ids(mesh::FileMesh{<:ExodusDatabase})
  return Exodus.read_ids(mesh.mesh_obj, Block)
end

function FiniteElementContainers.nodeset_ids(mesh::FileMesh{<:ExodusDatabase})
  return Exodus.read_ids(mesh.mesh_obj, NodeSet)
end

function FiniteElementContainers.sideset_ids(mesh::FileMesh{<:ExodusDatabase})
  return Exodus.read_ids(mesh.mesh_obj, SideSet)
end

function FiniteElementContainers.coordinates(mesh::FileMesh{<:ExodusDatabase}) 
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