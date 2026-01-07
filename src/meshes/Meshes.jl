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

const elem_type_map_2 = Dict{String, Type{<:ReferenceFiniteElements.AbstractElementType}}(
  # "HEX"     => Hex8,
  # "HEX8"    => Hex8,
  # only support on simple mesh types for now
  "QUAD"    => Quad,
  "QUAD4"   => Quad,
  # "QUAD9"   => Quad,
  # "TRI"     => Tri3,
  # "TRI3"    => Tri3,
  # "TRI6"    => Tri6,
  # "TET"     => Tet4,
  # "TETRA"   => Tet4,
  # "TETRA4"  => Tet4,
  # "TETRA10" => Tet10
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

@inline _canonical_edge(i, j) = i < j ? (i, j) : (j, i)

# coords coming in are nodal coords
# TODO currently assumes linear elements coming in
function _create_edges(conns)
  edge_dict = Dict{NTuple{2, Int}, Int}()
  edges = Dict{Symbol, Vector{NTuple{2, Int}}}()
  elem2edge = Dict{Symbol, Matrix{Int}}()
  edge2adjelem = Dict{Symbol, Vector{NTuple{2, Int}}}()
  for (key, conns) in pairs(conns)
    temp1, temp2, temp3 = _create_edges!(edge_dict, conns)
    edges[key] = temp1
    elem2edge[key] = temp2
    edge2adjelem[key] = temp3
  end
  
  edges = reduce(vcat, values(edges)) |> unique
  return edges, elem2edge, edge2adjelem
end

# TODO currently assumes tri3 elements
function _create_edges!(edge_dict, conns)
    @assert size(conns, 1) == 3 "Only Tri3 elements supported currently"
    edges = NTuple{2, Int}[]
    elem2edge = Matrix{Int}(undef, 3, size(conns, 2))
    edge2elem = NTuple{2, Int}[]

    for e in axes(conns, 2)
        n1, n2, n3 = conns[:, e]
        local_edges = (
            (n2, n3),
            (n3, n1),
            (n1, n2),
        )

        for le = 1:3
            key = _canonical_edge(local_edges[le]...)
        
            if haskey(edge_dict, key)
                edge_id = edge_dict[key]
                eL, eR = edge2elem[edge_id]
                @assert eR == -1  # no non-manifold edges
                edge2elem[edge_id] = (eL, e)
            else
                edge_id = length(edges) + 1
                edge_dict[key] = edge_id
                push!(edges, key)
                push!(edge2elem, (e, -1))
            end
        
            elem2edge[le, e] = edge_id
        end
    end
    return edges, elem2edge, edge2elem
end


include("AMRMesh.jl")
include("StructuredMesh.jl")
include("UnstructuredMesh.jl")