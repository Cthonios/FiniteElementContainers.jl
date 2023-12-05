abstract type AbstractMesh{Itype, Ftype} <: FEMContainer end
function copy_mesh end
function num_dimensions end
function num_nodes end
function coordinates end
function connectivity end
function element_type end

file_name(mesh::AbstractMesh) = mesh.file_name
int_type(::AbstractMesh{Itype, Ftype}) where {Itype, Ftype} = Itype
float_type(::AbstractMesh{Itype, Ftype}) where {Itype, Ftype} = Ftype

struct FileMesh{Itype, Ftype, MeshObj} <: AbstractMesh{Itype, Ftype}
  file_name::String
  mesh_obj::MeshObj
end
