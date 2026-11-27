"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
struct UnstructuredMesh{
  MeshObj,
  ND,
  RT <: Number,
  IT <: Integer
} <: AbstractMesh
  mesh_obj::MeshObj
  nodal_coords::H1Field{RT, Vector{RT}, ND}
  # element_block_names::Dict{IT, Symbol}
  element_block_names::Vector{String}
  element_block_names_map::Dict{IT, String}
  element_types::Dict{String, String}
  element_conns::Dict{String, Matrix{IT}}
  element_id_map::Vector{IT}
  element_id_maps::Dict{String, Vector{IT}}
  node_id_map::Vector{IT}
  nodeset_names::Dict{IT, String}
  nodeset_nodes::Dict{String, Vector{IT}}
  sideset_names::Dict{IT, String}
  sideset_elems::Dict{String, Vector{IT}}
  sideset_nodes::Dict{String, Vector{IT}}
  sideset_sides::Dict{String, Vector{IT}}
  sideset_side_nodes::Dict{String, Matrix{IT}}
end

"""
$(TYPEDSIGNATURES)
"""
function UnstructuredMesh(file_type::AbstractMeshFileType, file_name::String; kwargs...)
  file = FileMesh(file_type, file_name)
  mesh = UnstructuredMesh(file; kwargs...)
  finalize(file)
  return mesh
end

"""
$(TYPEDSIGNATURES)
TODO change the interface so we don't need
create_edges/create_faces. We should parse the 
interpolation/space type and determine this
change in behavior from these inputs...
"""
function UnstructuredMesh(
  file::FileMesh{T, V}; 
  create_edges::Bool                = false, 
  create_faces::Bool                = false,
  interp_type                       = Lagrange, # TODO further type me
  p_order::Union{Nothing, Int}      = nothing,
  space_type                        = H1Field                   
) where {T, V}
  # read nodal coords and ids
  nodal_coords, n_id_map = nodal_coordinates_and_ids(file)
  
  # read element block types, conn, etc.
  el_id_map = element_ids(file)
  el_conns, el_id_maps, el_block_names, el_block_names_map, el_types = element_blocks(file)

  # read nodesets
  nset_names, nset_nodes = nodesets(file)

  # read sidesets 
  sset_elems, sset_names, sset_nodes, sset_sides, sset_side_nodes = sidesets(file)

  mesh = UnstructuredMesh(
    file,
    nodal_coords, 
    el_block_names, el_block_names_map,
    el_types, el_conns, 
    el_id_map, el_id_maps, 
    n_id_map,
    nset_names, nset_nodes,
    sset_names, sset_elems, sset_nodes, 
    sset_sides, sset_side_nodes
  )
  return mesh
end

num_dimensions(mesh::UnstructuredMesh) = num_dimensions(mesh.mesh_obj)

include("Exodus.jl")

# dispatch based on file extension
"""
$(TYPEDSIGNATURES)
"""
function UnstructuredMesh(file_name::String; kwargs...)
  if !isfile(file_name)
    throw(ErrorException("Failed to find file $file_name"))
  end

  ext = splitext(file_name)
  if occursin(".geo", file_name) # special case where we generate mesh via gmsh
    return UnstructuredMesh(GmshMesh(), file_name; kwargs...) 
  elseif occursin(".g", file_name) || occursin(".e", file_name) || occursin(".exo", file_name)
    return UnstructuredMesh(ExodusMesh(), file_name; kwargs...)
  elseif occursin(".i", file_name) || occursin(".inp", file_name)
    # return UnstructuredMesh(AbaqusMesh(), file_name, create_edges, create_faces)
    @assert false "TODO finish me"
  elseif occursin(".msh", file_name)
    return UnstructuredMesh(GmshMesh(), file_name; kwargs...)
  else
    throw(ErrorException("Unsupported file type with extension $ext"))
  end
end

function copy_mesh(mesh::UnstructuredMesh, new_file::String)
  mt = mesh_type(mesh.mesh_obj)
  copy_mesh(mesh, new_file, mt)
  return nothing
end

# TODO write a show method
