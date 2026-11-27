const elem_type_map = Dict{String, Any}(
  "HEX"     => Hex,
  "HEX8"    => Hex,
  "QUAD"    => Quad,
  "QUAD4"   => Quad,
  "QUAD9"   => Quad,
  "TRI"     => Tri,
  "TRI3"    => Tri,
  "TRI6"    => Tri,
  "TET"     => Tet,
  "TETRA"   => Tet,
  "TETRA4"  => Tet,
  "TETRA10" => Tet,
  "TETRA15" => Tet
)

const elem_type_map_2 = Dict{String, Any}(
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

# different mesh file types
abstract type AbstractMeshFileType end
struct AbaqusMesh <: AbstractMeshFileType
end
struct ExodusMesh <: AbstractMeshFileType
end
struct GmshMesh <: AbstractMeshFileType
end

# minimum interface to define
function element_blocks end
function nodal_coordinates_and_ids end
function num_dimensions end

# additional optional helper interface
function nodal_coordinates end # TODO change to nodal
function nodesets end
function node_id_map end

"""
$(TYPEDEF)
"""
abstract type AbstractMesh end

"""
$(TYPEDSIGNATURES)
Element block names in canonical block order, i.e. the order the blocks appear
in the mesh file.

This is the single source of truth for what "block `b`" means. Meshes store
per-block data (`element_conns`, `element_id_maps`, `element_types`) in `Dict`s
keyed by block name, and `Dict` iteration order is hash order -- neither the
mesh file order nor stable under a change of block names. Anything that indexes
blocks by position must derive that order from here rather than from
`values(some_dict)`.
"""
block_names(mesh::AbstractMesh) = mesh.element_block_names

"""
$(TYPEDSIGNATURES)
Per-block element connectivities in canonical block order. See [`block_names`](@ref).
"""
block_conns(mesh::AbstractMesh) = [mesh.element_conns[name] for name in block_names(mesh)]

"""
$(TYPEDSIGNATURES)
Per-block element ID maps in canonical block order. See [`block_names`](@ref).
"""
block_id_maps(mesh::AbstractMesh) = [mesh.element_id_maps[name] for name in block_names(mesh)]

function Base.show(io::IO, mesh::AbstractMesh)
  println(io, typeof(mesh).name.name, ":")
  println(io, "  Number of dimensions = $(size(mesh.nodal_coords, 1))")
  println(io, "  Number of nodes      = $(size(mesh.nodal_coords, 2))")

  println(io, "  Element Blocks:")
  for name in block_names(mesh)
    conn = mesh.element_conns[name]
    println(io, "    $name:")
    println(io, "      Element type       = $(mesh.element_types[name])")
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
struct FileMesh{MeshObj, MeshType} <: AbstractMesh
  file_name::String
  mesh_obj::MeshObj

  function FileMesh{MeshObj, MeshType}(file_name::String, mesh_obj::MeshObj) where {MeshObj, MeshType}
    new{MeshObj, MeshType}(file_name, mesh_obj)
  end
end

mesh_type(::FileMesh{MeshObj, MeshType}) where {MeshObj, MeshType} = MeshType

# interface to define for FileMesh
function finalize end

function centroid(mesh::AbstractMesh)
  ND = size(mesh.nodal_coords, 1)
  xc = zero(SVector{ND, Float64})
  for n in axes(mesh.nodal_coords, 2)
    xc = xc + SVector{ND, Float64}(@views mesh.nodal_coords[:, n])
  end
  return xc / size(mesh.nodal_coords, 2)
end

function rigid_body_modes(mesh::AbstractMesh)
  xc = centroid(mesh)
  ND, NN = length(xc), size(mesh.nodal_coords, 2)
  if ND == 2
    modes = zeros(3, ND * NN)
    for (n, x) in enumerate(eachcol(mesh.nodal_coords))
      i = 2 * (n - 1)
      
      # translation modes
      modes[1, i + 1] = 1. # Tx
      modes[2, i + 2] = 1. # Ty

      # rotation
      # ω_z
      modes[3, i + 1] = -(x[2] - xc[2])
      modes[3, i + 2] = (x[1] - xc[1])
    end
  elseif ND == 3
    modes = zeros(6, ND * NN)
    for (n, x) in enumerate(eachcol(mesh.nodal_coords))
      i = 3 * (n - 1)
      dx = x - xc

      # translation modes
      modes[1, i + 1] = 1. # Tx
      modes[2, i + 2] = 1. # Ty
      modes[3, i + 3] = 1. # Tz

      # rotations

      # ω_x
      modes[4, i + 2] = -dx[3]
      modes[4, i + 3] = dx[2]

      # ω_y
      modes[5, i + 1] = dx[3]
      modes[5, i + 3] = -dx[1]

      # ω_z
      modes[6, i + 1] = -dx[2]
      modes[6, i + 2] = dx[1]
    end
  else
    modes = ones(ND, 1) # Tx
  end

  return modes
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
  # num_node_sets      = length(mesh.nodeset_nodes)
  num_side_sets      = 0
  num_node_sets      = 0

  # make init
  init = Initialization{Int32}(
    num_dim, num_nodes, num_elems,
    num_elem_blks, num_node_sets, num_side_sets
  )

  # create exo
  if isfile(file_name)
    rm(file_name; force = force)
  end
  exo = ExodusDatabase{Int32, Int32, Int32, eltype(mesh.nodal_coords)}(
    file_name, "w", init
  )

  # write coordinates
  coords = mesh.nodal_coords |> collect
  write_coordinates(exo, coords)

  # write node map
  # TODO
  # write_id_map(exo, NodeMap, convert.(Int32, mesh.node_id_map))

  # write block names
  write_names(exo, Exodus.Block, block_names(mesh))

  # TODO write block id maps
  # Blocks are written in ascending block id, matching the order the names were
  # just written in -- iterating the id => name `Dict` directly would write them
  # in hash order and pair each block with another block's name.
  for n in sort!(collect(keys(mesh.element_block_names_map)))
    block_name = mesh.element_block_names_map[n]
    el_type = mesh.element_types[block_name]
    conn = mesh.element_conns[block_name]
    write_block(exo, n, String(el_type), conn |> collect)
  end

  # TODO write nodesets
  # for (n, name) in mesh.nodeset_names
  #   nodes = mesh.nodeset_nodes[name]
  #   nset = NodeSet(Int32(n), convert.(Int32, nodes))
  #   write_set(exo, nset)
  # end
  # names = map(String, values(mesh.nodeset_names))
  # write_names(exo, NodeSet, names)

  # TODO write nodesets
  # for (n, name) in mesh.sideset_names
  #   elems = mesh.sideset_elems[name]
  #   nodes = mesh.sideset_nodes[name]
  #   sides = mesh.sideset_sides[name]
  # end

  close(exo)
end

include("StructuredMesh.jl")
include("Topology.jl")
include("UnstructuredMesh.jl")

include("Coloring.jl")