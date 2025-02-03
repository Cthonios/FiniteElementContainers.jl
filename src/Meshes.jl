"""
$(TYPEDEF)
"""
abstract type AbstractMesh <: FEMContainer end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function coordinates(::AbstractMesh) 
  @assert false
end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function copy_mesh end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function element_block_ids(::AbstractMesh) 
  @assert false
end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function element_block_names(::AbstractMesh) 
  @assert false
end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function element_connectivity(::AbstractMesh) 
  @assert false
end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function element_type(::AbstractMesh)
  @assert false
end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function nodeset(::AbstractMesh) 
  @assert false
end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function nodesets(::AbstractMesh, ids) 
  @assert false
end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function nodeset_ids(::AbstractMesh) 
  @assert false
end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function nodeset_names(::AbstractMesh) 
  @assert false
end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function sideset(::AbstractMesh)
  @assert false
end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function sideset_ids(::AbstractMesh) 
  @assert false
end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function sideset_names(::AbstractMesh) 
  @assert false
end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function num_dimensions(::AbstractMesh)
  @assert false
end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function num_nodes(::AbstractMesh) 
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

# new stuff below
struct UnstructuredMesh{X, ETypes, EConns, NSetNodes} <: AbstractMesh
  nodal_coords::X
  element_types::ETypes
  element_conns::EConns
  nodeset_nodes::NSetNodes
end

function UnstructuredMesh(file_type, file_name::String)
  file = FileMesh(file_type, file_name)

  # read nodal coordinates
  nodal_coords = coordinates(file)

  # read element block types, conn, etc.
  el_block_ids = element_block_ids(file)
  el_block_names = element_block_names(file)
  el_block_names = Symbol.(el_block_names)
  el_types = element_type.((file,), el_block_ids)
  el_types = NamedTuple{tuple(el_block_names...)}(tuple(el_types...))
  el_conns = element_connectivity.((file,), el_block_ids)
  el_conns = NamedTuple{tuple(el_block_names...)}(tuple(el_conns...))
  el_conns = ComponentArray(el_conns)

  # read nodesets
  nset_names = Symbol.(nodeset_names(file))
  nsets = nodesets(file, nodeset_ids(file))
  nset_nodes = NamedTuple{tuple(nset_names...)}(tuple(nsets...))
  nset_nodes = ComponentArray(nset_nodes)

  # read sidesets

  # TODO
  # write methods to create edge and face connectivity

  return UnstructuredMesh(nodal_coords, el_types, el_conns, nset_nodes)
end
