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
  "TETRA10" => Tet
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

function Base.show(io::IO, mesh::AbstractMesh)
  println(io, typeof(mesh).name.name, ":")
  println(io, "  Number of dimensions = $(size(mesh.nodal_coords, 1))")
  println(io, "  Number of nodes      = $(size(mesh.nodal_coords, 2))")

  println(io, "  Element Blocks:")
  for (name, type) in zip(values(mesh.element_block_names), mesh.element_types)
    conn = mesh.element_conns[name]
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

function create_higher_order_mesh(mesh::AbstractMesh, space_type, interp_type, p_degree::Int)
  @warn "This method assumes the mesh is linear. It currently does no error checking"
  @warn "This method still does not update nodesets appropriately"
  if p_degree == 1
    return mesh
  elseif p_degree == 0
    @assert false "TODO 0 order meshes"
  end

  if size(mesh.nodal_coords, 1) == 3
    @assert false "Currently 3D is not supported for this method"
  end

  # collect edges and filter out boundary edges
  edges = _create_linear_edges(mesh, space_type, interp_type)
  boundary_edges = filter(e -> length(e[2]) == 1, edges)

  # make higher order edges
  edge_interior_coords, edge_interior_nodes = _create_higher_order_edges(
    mesh, edges, space_type, interp_type, p_degree
  )

  # create new coords and conns
  # this will already have the vertex nodes
  new_coords = copy(mesh.nodal_coords)

  # add edge contributions to coords
  # need to do this before the interior method call
  append!(new_coords.data, edge_interior_coords)

  # add edge contributions to conns
  new_conns = Dict{Symbol, Matrix{Int}}()
  for (n, block_name) in mesh.element_block_names
    el_type = elem_type_map[String(mesh.element_types[block_name])]{Lagrange, p_degree}()
    el_type_linear = elem_type_map[String(mesh.element_types[block_name])]{Lagrange, 1}()
    NV = num_vertices_per_cell(el_type)
    ND = num_cell_dofs(el_type)
    conn = mesh.element_conns[block_name]
    temp_conn = Matrix{Int}(undef, ND, size(conn, 2))
    temp_conn[1:NV, :] .= conn

    for e in axes(conn, 2)
      local_edges = _create_local_edges(conn, el_type_linear, e)
      new_nodes = mapreduce(x -> edge_interior_nodes[_canonical_edge(x...)], vcat, local_edges)
      temp_conn[NV + 1:NV + length(new_nodes), e] .= new_nodes
    end
    new_conns[block_name] = temp_conn
  end

  # add interior contributions to coords and conns
  # these are element local
  _add_higher_order_interiors!(
    new_coords, new_conns, mesh.element_types, space_type, interp_type, p_degree
  )

  # now update the sideset nodes and sideset side nodes, 
  # all other sideset props shoudl stay the same
  new_sideset_nodes = Vector{Int}(undef, 0)
  new_sideset_side_nodes = Vector{Int}(undef, 0)
  for key in keys(mesh.sideset_side_nodes)
    
  end

  new_coords, new_conns
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
  # block_names = map(String, values(mesh.element_block_names))
  block_names = mesh.element_block_names
  write_names(exo, Block, block_names)

  # TODO write block id maps
  # for (n, block_name) in mesh.element_block_names
  for (n, block_name) in mesh.element_block_names_map
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

function _add_higher_order_interiors!(
  coords, conns, 
  el_types, ::Type{<:H1Field}, ::Type{<:Lagrange}, p_degree::Int
)
  n_nodes = size(coords, 2)
  for key in keys(conns)
    el_type = String(el_types[key])
    conn = conns[key]
    el_type = elem_type_map[el_type]{Lagrange, p_degree}()
    n_start = num_cell_dofs(el_type) - num_interior_dofs(el_type) + 1

    if n_start > num_cell_dofs(el_type)
      continue
    end

    for e in axes(conn, 2)
      temp_coords = _create_interior_nodes(coords, conn, el_type, e)
      for x in temp_coords
        append!(coords.data, x)
      end
      conn[n_start:end, e] = n_nodes + 1:n_nodes + length(temp_coords)
      n_nodes += length(temp_coords)
    end
  end
end

# edge stuff below
@inline _canonical_edge(i, j) = i < j ? (i, j) : (j, i)

# this is an H1 method, need to specialize
function _create_linear_edges(mesh::AbstractMesh, ::Type{<:H1Field}, ::Type{<:Lagrange})
  edge2elem = Dict{NTuple{2, Int}, Vector{NTuple{3, Int}}}()

  for (n, key) in mesh.element_block_names
    # el_type = mesh.element_types[n]
    el_type = mesh.element_types[key]
    @assert el_type == :QUAD || el_type == :QUAD4 ||
            el_type == :TRI  || el_type == :TRI3 "Unsupported element type $el_type"
    re = elem_type_map[String(el_type)]{Lagrange, 1}()
    conn = mesh.element_conns[key]
    el_ids = mesh.element_id_maps[key]
    for e in axes(conn, 2)
      local_edges = _create_local_edges(conn, re, e)
      el_id = el_ids[e]
      for (le_num, le) in enumerate(local_edges)
        ce = _canonical_edge(le...)
        orientation = le[1] < le[2] ? 1 : -1
        push!(
          get!(edge2elem, ce, Vector{NTuple{3, Int}}()),
          (el_id, le_num, orientation)
        )
      end
    end
  end
  return edge2elem
end

function _create_higher_order_edges(
  mesh, linear_edges, ::Type{<:H1Field}, ::Type{<:Lagrange}, p_degree::Int
)
  ND = size(mesh.nodal_coords, 1)
  coords = Vector{Float64}(undef, 0)
  edge_nodes = Dict{NTuple{2, Int}, Vector{Int}}()
  ξs = reverse(collect(1:p_degree - 1) ./ p_degree)
  # ξs .= 2 * (ξs .- 0.5)

  next_node_id = size(mesh.nodal_coords, 2)
  for edge in keys(linear_edges)
    x1 = SVector{ND, Float64}(@views mesh.nodal_coords[:, edge[1]])
    x2 = SVector{ND, Float64}(@views mesh.nodal_coords[:, edge[2]])
    node_ids = Vector{Int}(undef, p_degree - 1)

    for (k, ξ) in enumerate(ξs)
      # Linear interpolation
      # assuming edge goes from 0 to 1
      x = (1 - ξ) * x1 + ξ * x2

      # x = (1 - ξ) * x1 + (1 + ξ) * x2
      append!(coords, x)
      next_node_id += 1
      node_ids[k] = next_node_id
    end

    edge_nodes[edge] = node_ids
  end

  return coords, edge_nodes
end

function _create_interior_nodes(
  coords, conn, ::Quad{Lagrange, p}, e::Int
) where p
  x1 = SVector{2, Float64}(@views coords[:, conn[1, e]])
  x2 = SVector{2, Float64}(@views coords[:, conn[2, e]])
  x3 = SVector{2, Float64}(@views coords[:, conn[3, e]])
  x4 = SVector{2, Float64}(@views coords[:, conn[4, e]])

  ξs = reverse(collect(1:p - 1) ./ p)
  # ξs .= 2 * (ξs .- 0.5)
  new_coords = SVector{2, Float64}[]
  for η in ξs, ξ in ξs
    x = (1 - ξ) * (1 - η) * x1 +
        (0 + ξ) * (1 - η) * x2 +
        (0 + ξ) * (0 + η) * x3 +
        (1 - ξ) * (0 + η) * x4
    # x = (1 - ξ) * (1 - η) * x1 +
    #     (1 + ξ) * (1 - η) * x2 +
    #     (1 + ξ) * (1 + η) * x3 +
    #     (1 - ξ) * (1 + η) * x4
    push!(new_coords, x)
  end
  return new_coords
end

function _create_interior_nodes(
  coords, conn, ::Tri{Lagrange, p}, e::Int
) where p
  x1 = SVector{2, Float64}(@views coords[:, conn[1, e]])
  x2 = SVector{2, Float64}(@views coords[:, conn[2, e]])
  x3 = SVector{2, Float64}(@views coords[:, conn[3, e]])

  new_coords = Float64[]
  for i in 1:p - 2
    for j in 1:(p - 1 - i)
      k = p - i - j

      λ1 = i / p
      λ2 = j / p
      λ3 = k / p

      x = λ1 * x1 + λ2 * x2 + λ3 * x3
      append!(new_coords, x)
    end
  end
  return new_coords
end

# coords coming in are nodal coords
# TODO currently assumes linear elements coming in
function _create_linear_tri_edges(conns)
  edge_dict = Dict{NTuple{2, Int}, Int}()
  edges = Dict{Symbol, Vector{NTuple{2, Int}}}()
  elem2edge = Dict{Symbol, Matrix{Int}}()
  edge2adjelem = Dict{Symbol, Vector{NTuple{2, Int}}}()
  for (key, conns) in pairs(conns)
    temp1, temp2, temp3 = _create_linear_tri_edges!(edge_dict, conns)
    edges[key] = temp1
    elem2edge[key] = temp2
    edge2adjelem[key] = temp3
  end
  
  edges = reduce(vcat, values(edges)) |> unique
  return edges, elem2edge, edge2adjelem
end

# TODO currently assumes tri3 elements
function _create_linear_tri_edges!(edge_dict, conns)
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

function _create_local_edges(conn, ::Quad{Lagrange, 1}, e::Int)
  n1, n2, n3, n4 = conn[:, e]
  return (
    (n1, n2),
    (n2, n3),
    (n3, n4),
    (n4, n1)
  )
end

function _create_local_edges(conn, ::Tri{Lagrange, 1}, e::Int)
  n1, n2, n3 = conn[:, e]
  return (
    (n1, n2),
    (n2, n3),
    (n3, n1)
  )
end

include("AMRMesh.jl")
include("StructuredMesh.jl")
include("UnstructuredMesh.jl")