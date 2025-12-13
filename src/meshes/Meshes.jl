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

include("StructuredMesh.jl")
include("UnstructuredMesh.jl")