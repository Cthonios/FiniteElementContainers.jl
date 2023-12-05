module FiniteElementContainersExodusExt

using Exodus
using FiniteElementContainers

function FiniteElementContainers.FileMesh(::Type{ExodusDatabase}, file_name::String)
  exo = ExodusDatabase(file_name, "r")
  Itype, Ftype = Exodus.get_bulk_int_type(exo), Exodus.get_float_type(exo)
  return FileMesh{Itype, Ftype, typeof(exo)}(file_name, exo)
end

FiniteElementContainers.num_dimensions(mesh::FileMesh{Itype, Ftype, <:ExodusDatabase}) where {Itype, Ftype} = 
mesh.mesh_obj.init.num_dim
FiniteElementContainers.num_nodes(mesh::FileMesh{Itype, Ftype, <:ExodusDatabase}) where {Itype, Ftype} = 
mesh.mesh_obj.init.num_nodes

function FiniteElementContainers.coordinates(mesh::FileMesh{Itype, Ftype, <:ExodusDatabase}) where {Itype, Ftype}
  coords = Exodus.read_coordinates(mesh.mesh_obj)
  return coords
end

function FiniteElementContainers.connectivity(
  mesh::FileMesh{Itype, Ftype, <:ExodusDatabase},
  id::Integer
) where {Itype, Ftype}

  block = read_block(mesh.mesh_obj, id)
  return block.conn
end

function FiniteElementContainers.element_type(
  mesh::FileMesh{Itype, Ftype, <:ExodusDatabase},
  id::Integer
) where {Itype, Ftype}
  block = read_block(mesh.mesh_obj, id)
  return block.elem_type
end

function FiniteElementContainers.copy_mesh(file_1::String, file_2::String)
  Exodus.copy_mesh(file_1, file_2)
end

end # module