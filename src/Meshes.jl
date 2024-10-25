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
function num_nodes(mesh::AbstractMesh) 
  @assert false "This method needs to overriden in extensions!"
end

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

function create_structured_mesh_data(Nx, Ny, xExtent, yExtent)
  xs = LinRange(xExtent[1], xExtent[2], Nx)
  ys = LinRange(yExtent[1], yExtent[2], Ny)
  Ex = Nx - 1
  Ey = Ny - 1

  coords = Matrix{Float64}(undef, 2, Nx * Ny)
  n = 1
  for ny in 1:Ny
    for nx in 1:Nx
      coords[1, n] = xs[nx]
      coords[2, n] = ys[ny]
      n = n + 1
    end
  end

  conns = Matrix{Int64}(undef, 3, 2 * Ex * Ey)
  n = 1
  for ex in 1:Ex
    for ey in 1:Ey
      conns[1, n] = (ex - 1) + Nx * (ey - 1) + 1
      conns[2, n] = ex + Nx * (ey - 1) + 1
      conns[3, n] = ex + Nx * ey + 1
      conns[1, n + 1] = (ex - 1) + Nx * (ey - 1) + 1
      conns[2, n + 1] = ex + Nx * ey + 1
      conns[3, n + 1] = (ex - 1) + Nx * ey + 1
      n = n + 2
    end
  end
  return coords, conns
end