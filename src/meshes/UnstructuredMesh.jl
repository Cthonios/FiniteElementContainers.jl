"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
$(TYPEDFIELDS)
"""
struct UnstructuredMesh{
  MeshObj,
  ND,
  RT <: Number,
  IT <: Integer,
  EConns,
  EdgeConns,
  FaceConns
} <: AbstractMesh
  mesh_obj::MeshObj
  nodal_coords::H1Field{RT, Vector{RT}, ND}
  element_block_names::Vector{Symbol}
  element_types::Vector{Symbol}
  element_conns::EConns
  element_id_maps::Dict{Symbol, Vector{IT}}
  node_id_map::Vector{IT}
  nodeset_nodes::Dict{Symbol, Vector{IT}}
  sideset_elems::Dict{Symbol, Vector{IT}}
  sideset_nodes::Dict{Symbol, Vector{IT}}
  sideset_sides::Dict{Symbol, Vector{IT}}
  sideset_side_nodes::Dict{Symbol, Matrix{IT}}
  # new additions
  edge_conns::EdgeConns
  face_conns::FaceConns
end

"""
$(TYPEDSIGNATURES)
"""
function UnstructuredMesh(file_type::AbstractMeshType, file_name::String, create_edges::Bool, create_faces::Bool)
  file = FileMesh(file_type, file_name)
  return UnstructuredMesh(file, create_edges, create_faces)
end

function UnstructuredMesh(file::FileMesh{T}, create_edges::Bool, create_faces::Bool) where T

  # read nodal coordinates
  if num_dimensions(file) == 2
    coord_syms = (:X, :Y)
  elseif num_dimensions(file) == 3
    coord_syms = (:X, :Y, :Z)
  # else
  #   @assert false "Bad number of dimensions $(num_dimensions(file))"
  end

  nodal_coords = coordinates(file)
  nodal_coords = H1Field(nodal_coords)

  # read element block types, conn, etc.
  el_block_ids = element_block_ids(file)
  el_block_names = element_block_names(file)
  el_block_names = Symbol.(el_block_names)
  el_types = Symbol.(element_type.((file,), el_block_ids))
  el_conns = element_connectivity.((file,), el_block_ids)
  # el_conns = Dict(zip(el_block_names, el_conns))
  el_conns = map(Connectivity, el_conns)
  el_conns = NamedTuple{tuple(el_block_names...)}(tuple(el_conns...))
  el_id_maps = element_block_id_map.((file,), el_block_ids)
  el_id_maps = Dict(zip(el_block_names, el_id_maps))
  # el_id_maps = NamedTuple{tuple(el_block_names...)}(tuple(el_id_maps...))

  # read node id map
  n_id_map = convert.(Int64, node_id_map(file))

  # read nodesets
  nset_names = Symbol.(nodeset_names(file))
  nsets = nodesets(file, nodeset_ids(file))
  nset_nodes = Dict(zip(nset_names, nsets))

  # read sidesets 
  sset_names = Symbol.(sideset_names(file))
  ssets = sidesets(file, sideset_ids(file))

  sset_elems = Dict(zip(sset_names, map(x -> x[1], ssets)))
  sset_nodes = Dict(zip(sset_names, map(x -> x[2], ssets)))
  sset_sides = Dict(zip(sset_names, map(x -> x[3], ssets)))
  sset_side_nodes = Dict(zip(sset_names, map(x -> x[4], ssets)))


  # TODO also add edges/faces for sidesets, this may be tricky...

  # TODO
  # write methods to create edge and face connectivity
  # hack for now since we need a ref fe for this stuff
  #
  # TODO maybe move this into function space...

  if create_edges
    edges = ()
    all_edges = Dict{Tuple{Vararg{Int}}, Int}()

    for n in 1:length(el_types)
      ref_fe = ReferenceFE(elem_type_map[el_types[n]]{Lagrange, 1}())
      block_edges = Vector{SVector{length(ref_fe.edge_nodes), Int}}(undef, 0)
      _create_edges!(all_edges, block_edges, values(el_conns)[n], ref_fe)
      edges = (edges..., Connectivity{length(block_edges[1]), length(block_edges)}(reduce(vcat, block_edges)))
      # TODO need to create an id map
    end

    edges = NamedTuple{keys(el_conns)}(edges)
  else
    edges = nothing
  end

  # TODO need to finish this up
  if create_faces
    @assert num_dimensions(file) > 2 "Faces require a 3D mesh"
    @assert false "Need to fix this..."
    # faces = ()
    # all_faces = Dict{Tuple{Vararg{Int}}, Int}()

    # for n in 1:length(el_types)
    #   ref_fe = ReferenceFE(elem_type_map[el_types[n]]{Lagrange, 1}())
    #   block_faces = Vector{SVector{length(ref_fe.face_nodes), Int}}(undef, 0)
    #   _create_faces!(all_faces, block_faces, values(el_conns)[n], ref_fe)
    #   faces = (faces..., block_faces)
    #   # TODO need to create an id map
    # end
    # @show length(all_faces)
    # faces = NamedTuple{keys(el_conns)}(faces)
  else
    faces = nothing
  end

  return UnstructuredMesh(
    file,
    nodal_coords, 
    el_block_names, el_types, el_conns, el_id_maps, 
    n_id_map,
    nset_nodes,
    sset_elems, 
    sset_nodes, 
    sset_sides, sset_side_nodes,
    edges, faces
  )
end

num_dimensions(mesh::UnstructuredMesh) = num_dimensions(mesh.mesh_obj)

function _mesh_file_type end

include("Exodus.jl")

# dispatch based on file extension
"""
$(TYPEDSIGNATURES)
"""
function UnstructuredMesh(file_name::String; create_edges=false, create_faces=false)
  ext = splitext(file_name)
  if occursin(".g", file_name) || occursin(".e", file_name) || occursin(".exo", file_name)
    return UnstructuredMesh(ExodusMesh(), file_name, create_edges, create_faces)
  elseif occursin(".i", file_name) || occursin(".inp", file_name)
    return UnstructuredMesh(AbaqusMesh(), file_name, create_edges, create_faces)
  else
    throw(ErrorException("Unsupported file type with extension $ext"))
  end
end

# TODO write a show method
