abstract type AbstractMesh <: FEMContainer end

function coordinates end
function copy_mesh end
function element_block_ids end
function element_connectivity end
function element_type end
function nodeset end
function nodeset_ids end
function sideset end
function sideset_ids end
function num_dimensions end
function num_nodes end

"""
Returns file name for an mesh type
"""
file_name(mesh::AbstractMesh) = mesh.file_name

"""
Mesh type that has a handle to an open mesh file object.
This type's methods are "overridden" in extensions.

See FiniteElementContainersExodusExt for an example.
"""
struct FileMesh{MeshObj} <: AbstractMesh
  file_name::String
  mesh_obj::MeshObj
end

"""
Mesh type that should have most everything one might want.

TODO probably need some type of face/edge sets as well
TODO make some of these things optionals
"""
struct Mesh{
  D, 
  Coords      <: NodalField,
  BlockIds    <: AbstractArray{<:Integer, 1},
  Conns       <: NamedTuple,
  ElemTypes   <: NamedTuple,
  NsetIds     <: AbstractArray{<:Integer, 1},
  NsetNodes   <: AbstractArray{<:AbstractArray{<:Integer, 1}, 1},
  SsetIds     <: AbstractArray{<:Integer, 1},
  SsetElems   <: AbstractArray{<:AbstractArray{<:Integer, 1}, 1},
  SsetSides   <: AbstractArray{<:AbstractArray{<:Integer, 1}, 1}
} <: AbstractMesh

  file_name::String
  coords::Coords
  block_ids::BlockIds
  conns::Conns
  elem_types::ElemTypes
  nset_ids::NsetIds
  nset_nodes::NsetNodes
  sset_ids::SsetIds
  sset_elems::SsetElems
  sset_sides::SsetSides
end

"""
Mesh constructor
"""
function Mesh(
  type::Type, file_name::String;
  block_ids = nothing,
  nset_ids  = nothing,
  sset_ids  = nothing
)

  f = FileMesh(type, file_name)

  # defaults - set to all blocks, nsets, and ssets
  if block_ids === nothing
    block_ids = element_block_ids(f)
  end

  if nset_ids === nothing
    nset_ids = nodeset_ids(f)
  end

  if sset_ids === nothing
    sset_ids = sideset_ids(f)
  end

  # read coordinates
  coords = coordinates(f)
  coords = NodalField{size(coords, 1), size(coords, 2)}(coords, :nodal_X)
  # coords = Noda

  # read block connectivity and element types
  # conns            = Matrix{Int64}[]
  # conns            = Connectivity[]
  # elem_types       = String[]
  conns      = (;)
  elem_types = (;)
  ids_in_mesh = element_block_ids(f)
  # TODO a few cleanups are needed below
  for (n, id) in enumerate(block_ids)
    @assert id in ids_in_mesh "Block id $id not found in mesh $file_name"
    temp = element_connectivity(f, id)
    temp = convert.(Int64, temp)
    # N = size(temp, 1)
    # temp = reinterpret(SVector{N, Int64}, vec(temp)) |> collect
    # temp = StructArray(temp)
    # TODO, if we want to make connectivity static, we do it here!
    conn = Connectivity{size(temp, 1), size(temp, 2)}(temp, id |> Int64) # TODO clean up Int32 vs. Int64 stuff

    # TODO add block names 
    field_name = Symbol("connectivity_id_$id")

    conns      = (; conns..., field_name => conn)
    elem_types = (; elem_types..., field_name => element_type(f, id))
  end

  # read nodesets
  nset_nodes  = Vector{Int64}[]
  ids_in_mesh = nodeset_ids(f)
  for id in nset_ids
    @assert id in ids_in_mesh "NodeSet id $id not found in mesh $file_name"
    push!(nset_nodes, nodeset(f, id))
  end

  # read sidesets
  sset_elems = Vector{Int64}[]
  sset_sides = Vector{Int64}[]
  ids_in_mesh = sideset_ids(f)
  for id in sset_ids
    @assert id in ids_in_mesh "SideSet in $id not found in mesh $file_name"
    elems, sides = sideset(f, id)
    push!(sset_elems, elems)
    push!(sset_sides, sides)
  end

  # TODO cleanup make everything int64
  block_ids = convert.(Int64, block_ids)
  nset_ids  = convert.(Int64, nset_ids)
  sset_ids  = convert.(Int64, sset_ids)

  return Mesh{
    convert(Int64, num_dimensions(f)), # TODO another cleanup to be done
    typeof(coords),
    typeof(block_ids), typeof(conns), typeof(elem_types),
    typeof(nset_ids), typeof(nset_nodes),
    typeof(sset_ids), typeof(sset_elems), typeof(sset_sides)
  }(
    file_name,
    coords, 
    block_ids, conns, elem_types, 
    nset_ids, nset_nodes, 
    sset_ids, sset_elems, sset_sides
  )
end

connectivity(mesh::Mesh, block_index::Int)         = mesh.conns[block_index]
connectivity(mesh::Mesh, block_index::Int, e::Int) = connectivity(mesh.conns[block_index], e)
coordinates(mesh::Mesh)                            = mesh.coords

num_dimensions(
  ::Mesh{
    D,
    # D, NNodes, 
    Coords, 
    BlockIds, Conns, ElemTypes, 
    NsetIds, NsetNodes, 
    SsetIDs, SsetElems, SsetSides
  }
) where {D, Coords, BlockIds, Conns, ElemTypes,
  NsetIds, NsetNodes, SsetIDs, SsetElems, SsetSides} = D


# """
# In place method for collecting element level coordinates
#   TODO need to add a method that takes into account element coloring maybe?
# """
# function element_level_coordinates!(el_coords, mesh::Mesh, conn::Connectivity)
#   D         = num_dimensions(mesh)
#   NN        = num_nodes_per_element(conn)
#   coords = coordinates(mesh)
#   for e in axes(el_coords, 1)
#     el_coords[e] = SMatrix{D, NN, eltype(coords), D * NN}(vec(@views coords[:, connectivity(conn, e)]))
#   end 
# end

# """
# Wrapper for type stable in place by index
# """
# function element_level_coordinates!(Xs, mesh::Mesh, block_index::Int)
#   conn = connectivity(mesh, block_index)
#   element_level_coordinates!(Xs, mesh, conn)
# end

# """
# Out of place method for collecting element level coordinates
# This simply calls element_level_coordinates! with a pre-allocated
# ElementField. Currently it uses a structarray
# """
# function element_level_coordinates(mesh::Mesh, conn::Connectivity)
#   D         = num_dimensions(mesh)
#   NN        = num_nodes_per_element(conn)
#   T         = SMatrix{D, NN, eltype(coordinates(mesh)), D * NN}
#   names     = :element_X
#   el_coords = ElementField{D * NN, num_elements(conn)}(StructArray, T, names, undef)
#   element_level_coordinates!(el_coords, mesh, conn)
#   return el_coords
# end

# """
# Wrapper to prevent type instabilities on different Connectivity types
# """
# function element_level_coordinates(mesh::Mesh, block_index::Int)
#   conn      = connectivity(mesh, block_index)
#   el_coords = element_level_coordinates(mesh, conn)
#   return el_coords
# end

# # need this method for type stability 
# function element_level_coordinates(mesh::Mesh, conn::Connectivity, e::Int)
#   D         = num_dimensions(mesh)
#   NN        = num_nodes_per_element(conn)
#   coords    = coordinates(mesh)
#   el_coords = SMatrix{D, NN, eltype(coords), D * NN}(vec(@views coords[:, connectivity(conn, e)]))
#   return el_coords
# end

# # need this wrapper to eliminate most type instabilities
# # TODO real question is how to store these differently typed arrays?
# function element_level_coordinates(mesh::Mesh, block_index::Int, e::Int)
#   conn      = connectivity(mesh, block_index)
#   el_coords = element_level_coordinates(mesh, conn, e)
#   return el_coords
# end

# # need this method for type stability
# """
# A method that returns a ReinterpretArray for quicker access
# """
# function element_level_coordinates_reinterpret(mesh::Mesh, conn::Connectivity)
#   D         = num_dimensions(mesh)
#   NN        = num_nodes_per_element(conn)
#   coords    = coordinates(mesh)
#   el_coords = reinterpret(SMatrix{D, NN, eltype(coords), D * NN}, vec(@views coords[:, connectivity(conn)]))
#   return el_coords
# end

# # need this wrapper to eliminate most type instabilities
# # TODO real question is how to store these differently typed arrays?
# """
# Wrapper to prevent type instabilities
# """
# function element_level_coordinates_reinterpret(mesh::Mesh, block_index::Int)
#   conn      = connectivity(mesh, block_index)
#   el_coords = element_level_coordinates_reinterpret(mesh, conn)
#   return el_coords
# end

###############################################################################################3







# TODO move these to a more sensible place


"""
In place method for collecting element level fields
  TODO need to add a method that takes into account element coloring maybe?
"""
function element_level_fields!(u_els, conn::Connectivity, u::NodalField)
  NFields = num_fields(u)
  NN      = num_nodes_per_element(conn)
  for e in axes(u_els, 1)
    u_els[e] = SMatrix{NFields, NN, eltype(u), NFields * NN}(vec(@views u[:, connectivity(conn, e)]))
  end 
end

"""
Wrapper for type stable in place by index
"""
function element_level_fields!(u_els, mesh::Mesh, block_index::Int, u::NodalField)
  conn = connectivity(mesh, block_index)
  element_level_fields!(u_els, conn, u)
end

"""
Out of place method for collecting element level fields
This simply calls element_level_fields! with a pre-allocated
ElementField. Currently it uses a structarray
"""
function element_level_fields(conn::Connectivity, u::NodalField)
  NFields = num_fields(u)
  NN      = num_nodes_per_element(conn)
  T       = SMatrix{NFields, NN, eltype(u), NFields * NN}
  names   = Symbol("element_", field_names(u))
  # u_els   = ElementField{NFields * NN, num_elements(conn)}(StructArray, T, names, undef)
  u_els   = ElementField{NFields * NN, num_elements(conn), StructArray, T}(undef, names)
  element_level_fields!(u_els, conn, u)
  return u_els
end

"""
Wrapper to prevent type instabilities on different Connectivity types
"""
function element_level_fields(mesh::Mesh, block_index::Int, u::NodalField)
  conn = connectivity(mesh, block_index)
  u_el = element_level_fields(conn, u)
  return u_el
end

# need this method for type stability 
function element_level_fields(conn::Connectivity, e::Int, u::NodalField)
  NFields = num_fields(u)
  NN      = num_nodes_per_element(conn)
  u_els   = SMatrix{NFields, NN, eltype(u), NFields * NN}(vec(@views u[:, connectivity(conn, e)]))
  return u_els
end

# need this wrapper to eliminate most type instabilities
# TODO real question is how to store these differently typed arrays?
function element_level_fields(mesh::Mesh, block_index::Int, e::Int, u::NodalField)
  conn  = connectivity(mesh, block_index)
  u_els = element_level_fields(conn, e, u)
  return u_els
end

# need this method for type stability
"""
A method that returns a ReinterpretArray for quicker access
"""
function element_level_fields_reinterpret(conn::Connectivity, u::NodalField)
  NFields = num_fields(u)
  NN      = num_nodes_per_element(conn)
  u_els   = reinterpret(SMatrix{NFields, NN, eltype(u), NFields * NN}, vec(@views u[:, connectivity(conn)]))
  return u_els
end

# need this wrapper to eliminate most type instabilities
# TODO real question is how to store these differently typed arrays?
"""
Wrapper to prevent type instabilities
"""
function element_level_fields_reinterpret(mesh::Mesh, block_index::Int, u::NodalField)
  conn  = connectivity(mesh, block_index)
  u_els = element_level_fields_reinterpret(conn, u)
  return u_els
end


#################################################
# wrapper for coordinates
element_level_coordinates!(X_els, mesh::Mesh, block_index::Int) = element_level_fields!(X_els, mesh, block_index, mesh.coords)
element_level_coordinates(mesh::Mesh, block_index::Int)         = element_level_fields(mesh, block_index, mesh.coords)
element_level_coordinates(mesh::Mesh, block_index::Int, e::Int) = element_level_fields(mesh, block_index, e, mesh.coords)
