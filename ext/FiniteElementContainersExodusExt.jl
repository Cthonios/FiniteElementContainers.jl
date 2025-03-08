module FiniteElementContainersExodusExt

using DocStringExtensions
using Exodus
using FiniteElementContainers

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
function FiniteElementContainers.FileMesh(::Type{<:FiniteElementContainers.ExodusMesh}, file_name::String)
  exo = ExodusDatabase(file_name, "r")
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
function FiniteElementContainers.element_block_id_map(mesh::FileMesh{<:ExodusDatabase}, id)
  return convert.(Int64, Exodus.read_block_id_map(mesh.mesh_obj, id))
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
function FiniteElementContainers.element_block_names(mesh::FileMesh{<:ExodusDatabase})
  return Exodus.read_names(mesh.mesh_obj, Block)
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
function FiniteElementContainers.nodeset_names(mesh::FileMesh{<:ExodusDatabase})
  return Exodus.read_names(mesh.mesh_obj, NodeSet)
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
function FiniteElementContainers.sideset_names(mesh::FileMesh{<:ExodusDatabase})
  return Exodus.read_names(mesh.mesh_obj, SideSet)
end

"""
$(TYPEDSIGNATURES)
"""
function FiniteElementContainers.coordinates(mesh::FileMesh{ExodusDatabase{M, I, B, F, Init}})::Matrix{F} where {M, I, B, F, Init} 
  coords = Exodus.read_coordinates(mesh.mesh_obj)
  return coords
end

function FiniteElementContainers.element_connectivity(
  mesh::FileMesh{<:ExodusDatabase},
  id::Integer
) 

  block = read_block(mesh.mesh_obj, id)
  return convert.(Int64, block.conn)
end

function FiniteElementContainers.element_connectivity(
  mesh::FileMesh{<:ExodusDatabase},
  name::String
) 

  block = read_block(mesh.mesh_obj, name)
  return block.conn
end


function FiniteElementContainers.element_type(
  mesh::FileMesh{<:ExodusDatabase},
  id::Integer
) 
  block = read_block(mesh.mesh_obj, id)
  return block.elem_type
end

function FiniteElementContainers.element_type(
  mesh::FileMesh{<:ExodusDatabase},
  name::String
) 
  block = read_block(mesh.mesh_obj, name)
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
  return sort!(convert.(Int64, nset.nodes))
end

function FiniteElementContainers.nodesets(
  mesh::FileMesh{<:ExodusDatabase},
  ids
) 
  return FiniteElementContainers.nodeset.((mesh,), ids)
end

function FiniteElementContainers.sideset(
  mesh::FileMesh{<:ExodusDatabase},
  id::Integer
) 
  sset = read_set(mesh.mesh_obj, SideSet, id)
  elems = convert.(Int64, sset.elements)
  sides = convert.(Int64, sset.sides)
  nodes = convert.(Int64, Exodus.read_side_set_node_list(mesh.mesh_obj, id)[2])
  unique!(sort!(nodes))
  perm = sortperm(elems)
  return elems[perm], nodes, sides[perm]
end

function FiniteElementContainers.sidesets(
  mesh::FileMesh{<:ExodusDatabase},
  ids
) 
  return FiniteElementContainers.sideset.((mesh,), ids)
end

end # module
