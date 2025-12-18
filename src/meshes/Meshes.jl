const elem_type_map = Dict{String, Type{<:ReferenceFiniteElements.AbstractElementType}}(
  "HEX"     => Hex8,
  "HEX8"    => Hex8,
  "QUAD"    => Quad4,
  "QUAD4"   => Quad4,
  "QUAD9"   => Quad9,
  "TRI"     => Tri3,
  "TRI3"    => Tri3,
  "TRI6"    => Tri6,
  "TET"     => Tet4,
  "TETRA"   => Tet4,
  "TETRA4"  => Tet4,
  "TETRA10" => Tet10
)

# different mesh types
abstract type AbstractMeshType end
struct AbaqusMesh <: AbstractMeshType
end
struct ExodusMesh <: AbstractMeshType
end

"""
$(TYPEDEF)
"""
abstract type AbstractMesh end

function Base.show(io::IO, mesh::AbstractMesh)
  println(io, typeof(mesh).name.name, ":")
  println(io, "  Number of dimensions = $(size(mesh.nodal_coords, 1))")
  println(io, "  Number of nodes      = $(size(mesh.nodal_coords, 2))")

  println(io, "  Element Blocks:")
  for (conn, name, type) in zip(mesh.element_conns, mesh.element_block_names, mesh.element_types)
    println(io, "    $name:")
    println(io, "      Element type       = $type")
    println(io, "      Number of elements = $(size(conn, 2))")
  end

  println(io, "  Node sets:")
  for (name, nodes) in mesh.nodeset_nodes
    println(io, "    $name:")
    println(io, "      Number of nodes = $(length(nodes))")
  end

  println(io, "  Side sets:")
  for (name, sides) in mesh.sideset_sides
    println(io, "    $name:")
    println(io, "      Number of elements = $(length(sides))")
  end
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

# TODO might need to be careful about int types below
function write_to_file(mesh::AbstractMesh, file_name::String; force::Bool = false)
  if force && isfile(file_name)
    Base.rm(file_name; force = true)
  end

  # initialization parameters
  num_dim, num_nodes = size(mesh.nodal_coords)
  num_elems          = mapreduce(x -> size(x, 2), +, values(mesh.element_conns))
  num_elem_blks      = length(mesh.element_conns)
  # num_side_sets      = length(mesh.sideset_elems)
  num_node_sets      = length(mesh.nodeset_nodes)
  num_side_sets      = 0
  # num_node_sets      = 0

  # make init
  init = Initialization{Int32}(
    num_dim, num_nodes, num_elems,
    num_elem_blks, num_node_sets, num_side_sets
  )

  # create exo
  exo = ExodusDatabase{Int32, Int32, Int32, eltype(mesh.nodal_coords)}(
    file_name, "w", init
  )

  # write coordinates
  coords = mesh.nodal_coords |> collect
  write_coordinates(exo, coords)

  # write node map
  write_id_map(exo, NodeMap, convert.(Int32, mesh.node_id_map))

  # write block names
  write_names(exo, Block, map(String, mesh.element_block_names))

  # TODO write block id maps
  for n in axes(mesh.element_block_names, 1)
    write_block(exo, n, String(mesh.element_types[n]), values(mesh.element_conns)[n] |> collect)
  end

  # write nodesets
  names = keys(mesh.nodeset_nodes) |> collect
  for (n, name) in enumerate(names)
    nodes = mesh.nodeset_nodes[name]
    nset = NodeSet(Int32(n), convert.(Int32, nodes))
    write_set(exo, nset)
  end
  names = map(String, names)
  write_names(exo, NodeSet, names)

  # TODO write nodesets
  names = keys(mesh.sideset_elems) |> collect
  for (n, name) in enumerate(names)
    elems = mesh.sideset_elems[name]
    nodes = mesh.sideset_nodes[name]
    sides = mesh.sideset_sides[name]
  end

  close(exo)
end

include("StructuredMesh.jl")
include("UnstructuredMesh.jl")