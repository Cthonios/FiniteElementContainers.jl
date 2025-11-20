elem_type_map = Dict{String, Type{<:ReferenceFiniteElements.AbstractElementType}}(
  "HEX"     => Hex8,
  "HEX8"    => Hex8,
  "QUAD"    => Quad4,
  "QUAD4"   => Quad4,
  "QUAD9"   => Quad9,
  "TRI"     => Tri3,
  "TRI3"    => Tri3,
  "TRI6"    => Tri6,
  "TET"     => Tet4,
  "TETRA4"  => Tet4,
  "TETRA10" => Tet10
)

"""
$(TYPEDEF)
"""
abstract type AbstractMesh end
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
function element_block_id_map(::AbstractMesh, id) 
  @assert false
end
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
function element_connectivity(::AbstractMesh, id) 
  @assert false
end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function element_type(::AbstractMesh, id)
  @assert false
end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function node_cmaps(::AbstractMesh, rank) 
  @assert false
end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function node_id_map(::AbstractMesh) 
  @assert false
end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function nodeset(::AbstractMesh, id) 
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
function sideset(::AbstractMesh, id)
  @assert false
end
"""
$(TYPEDSIGNATURES)
Dummy method to be overriden for specific mesh file format
"""
function sidesets(::AbstractMesh) 
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

# TODO change to a type that subtypes AbstractMesh
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

function _create_edges!(all_edges, block_edges, el_to_nodes, ref_fe)
  local_edge_to_nodes = ref_fe.edge_nodes

  # loop over all elements connectivity
  for e in axes(el_to_nodes, 2)
    el_nodes = el_to_nodes[:, e]
    el_edges = map(x -> el_nodes[x], local_edge_to_nodes)
    # loop over edges
    local_edges = ()
    for edge in el_edges
      sorted_edge = sort(edge).data
      if !haskey(all_edges, sorted_edge)
        all_edges[sorted_edge] = length(all_edges) + 1
        local_edges = (local_edges..., length(all_edges) + 1)
      else
        local_edges = (local_edges..., all_edges[sorted_edge])
      end
    end
    push!(block_edges, local_edges)
  end
  return nothing
end

# this one isn't quite working
# function _create_faces!(all_faces, block_faces, el_to_nodes, ref_fe)
#   local_face_to_nodes = ref_fe.face_nodes
#   # display(local_face_to_nodes)
#   # loop over all elements connectivity
#   for e in axes(el_to_nodes, 2)
#     el_nodes = el_to_nodes[:, e]
#     el_faces = map(x -> el_nodes[x], local_face_to_nodes)
#     # display(el_faces)
#     # loop over faces
#     local_faces = ()
#     for face in el_faces
#       sorted_face = sort(face).data
#       if !haskey(all_faces, sorted_face)
#         all_faces[sorted_face] = length(all_faces)
#         local_faces = (local_faces..., length(all_faces))
#       else
#         local_faces = (local_faces..., all_faces[sorted_face])
#       end
#     end
#     push!(block_faces, local_faces)
#   end
#   return nothing
# end

# new stuff below
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
function UnstructuredMesh(file_type, file_name::String, create_edges::Bool, create_faces::Bool)
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

# different mesh types
abstract type AbstractMeshType end
struct ExodusMesh <: AbstractMeshType
end

function _mesh_file_type end

# dispatch based on file extension
"""
$(TYPEDSIGNATURES)
"""
function UnstructuredMesh(file_name::String; create_edges=false, create_faces=false)
  ext = splitext(file_name)
  if occursin(".g", file_name) || occursin(".e", file_name) || occursin(".exo", file_name)
    return UnstructuredMesh(ExodusMesh, file_name, create_edges, create_faces)
  else
    throw(ErrorException("Unsupported file type with extension $ext"))
  end
end

# TODO write a show method
