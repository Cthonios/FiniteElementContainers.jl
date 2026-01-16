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
  EdgeConns,
  FaceConns
} <: AbstractMesh
  mesh_obj::MeshObj
  nodal_coords::H1Field{RT, Vector{RT}, ND}
  element_block_names::Dict{IT, Symbol}
  element_types::Dict{Symbol, Symbol}
  element_conns::Dict{Symbol, Matrix{IT}}
  element_id_maps::Dict{Symbol, Vector{IT}}
  node_id_map::Vector{IT}
  nodeset_names::Dict{IT, Symbol}
  nodeset_nodes::Dict{Symbol, Vector{IT}}
  sideset_names::Dict{Int, Symbol}
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
  file::FileMesh{T}; 
  create_edges::Bool                = false, 
  create_faces::Bool                = false,
  interp_type                       = Lagrange, # TODO further type me
  p_order::Union{Nothing, Int}      = nothing,
  space_type                        = H1Field                   
) where T
  # read nodal coords and ids
  nodal_coords, n_id_map = nodal_coordinates_and_ids(file)
  
  # read element block types, conn, etc.
  el_conns, el_id_maps, el_block_names, el_types = element_blocks(file)

  # read nodesets
  nset_names, nset_nodes = nodesets(file)

  # read sidesets 
  sset_elems, sset_names, sset_nodes, sset_sides, sset_side_nodes = sidesets(file)

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

  mesh = UnstructuredMesh(
    file,
    nodal_coords, 
    el_block_names, el_types, el_conns, el_id_maps, 
    n_id_map,
    nset_names, nset_nodes,
    sset_names, sset_elems, sset_nodes, 
    sset_sides, sset_side_nodes,
    edges, faces
  )

  if p_order !== nothing
    if p_order == 1
      return mesh
    end
    @assert space_type == H1Field && interp_type == Lagrange
    coords, conns = create_higher_order_mesh(mesh, space_type, interp_type, p_order)
    el_types = Dict{Symbol, Symbol}()
    for (key, el_type) in mesh.element_types
      if el_type == :QUAD || el_type == :QUAD4
        if p_order == 2
          el_types[key] = :QUAD9
        else
          @assert false "TODO"
        end
      elseif el_type == :TRI || el_type == :TRI3
        if p_order == 2
          el_types[key] = :TRI6
        else
          @assert false "TODO"
        end
      else
        @assert false "Unsupported element or TODO $el_type"
      end
    end
    return UnstructuredMesh(
      file,
      coords, 
      mesh.element_block_names, el_types, conns, mesh.element_id_maps, 
      # mesh.node_id_map,
      1:size(coords, 2) |> collect,
      mesh.nodeset_names, 
      # mesh.nset_nodes,
      # mesh.sset_names, mesh.sset_elems, mesh.sset_nodes, 
      # mesh.sset_sides, mesh.sset_side_nodes,
      # edges, faces
      Dict{Symbol, Vector{Int}}(),
      mesh.sideset_names, mesh.sideset_elems, Dict{Symbol, Vector{Int}}(),
      mesh.sideset_sides, Dict{Symbol, Matrix{Int}}(),
      nothing, nothing
    )
  else
    return mesh
  end
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
