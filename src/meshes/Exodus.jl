# TODO make this mesh agnostic
function copy_mesh(mesh, file_2::String, ::Type{ExodusMesh})
  file_1 = mesh.mesh_obj.file_name
  Exodus.copy_mesh(file_1, file_2)
  return nothing
end

function copy_mesh(mesh, file_2::String, ::Type{ExodusMesh}, exo_type::Type{ExodusDatabase{M, I, B, F}}) where {M, I, B, F}
  file_1 = mesh.mesh_obj.file_name
  Exodus.copy_mesh(exo_type, file_1, file_2)
  return nothing
end

"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
function FileMesh(::ExodusMesh, file_name::String)
  exo = ExodusDatabase(file_name, "r")
  return FileMesh{typeof(exo), ExodusMesh}(file_name, exo)
end

function finalize(::FileMesh{MO, ExodusMesh}) where MO
  # Exodus.close(file.mesh_obj)
  return nothing
end

# minimum interface
function element_blocks(mesh::FileMesh{<:ExodusDatabase, ExodusMesh})
  blocks = read_sets(mesh.mesh_obj, Exodus.Block)
  block_ids = convert(Vector{Int}, map(x -> x.id, blocks))
  conns = map(x -> convert(Matrix{Int}, x.conn), blocks)
  # el_id_maps = element_block_id_map.((mesh,), block_ids)
  el_id_maps = map(x -> convert(Vector{Int}, Exodus.read_block_id_map(mesh.mesh_obj, x)), block_ids)
  names = Exodus.read_names(mesh.mesh_obj, Exodus.Block)
  types = map(x -> x.elem_type, blocks)

  conns = Dict(zip(names, conns))
  el_id_maps = Dict(zip(names, el_id_maps))
  types = Dict(zip(names, types))
  names_map = Dict(zip(block_ids, names))

  return conns, el_id_maps, names, names_map, types
end

function element_ids(mesh::FileMesh{<:ExodusDatabase, ExodusMesh})
  el_id_map = convert(Vector{Int}, Exodus.read_id_map(mesh.mesh_obj, ElementMap))
  return el_id_map
end

"""
$(TYPEDSIGNATURES)
"""
function nodal_coordinates_and_ids(mesh::FileMesh{<:ExodusDatabase, ExodusMesh})
  return nodal_coordinates(mesh), node_id_map(mesh)
end

function nodal_coordinates_and_ids(type::Type{<:H1Field}, mesh::FileMesh{<:ExodusDatabase, ExodusMesh})
  return nodal_coordinates(type, mesh), node_id_map(mesh)
end

function nodesets(mesh::FileMesh{<:ExodusDatabase, ExodusMesh})
  ids = convert(Vector{Int}, Exodus.read_ids(mesh.mesh_obj, NodeSet))
  nsets = Exodus.read_sets(mesh.mesh_obj, NodeSet)
  names = Exodus.read_names(mesh.mesh_obj, NodeSet)
  nodes = map(x -> sort(convert(Vector{Int}, x.nodes)), nsets)
  nodes = Dict{String, Vector{Int}}(zip(names, nodes))
  names = Dict{Int, String}(zip(ids, names))
  return names, nodes
end

function num_dimensions(
  mesh::FileMesh{<:ExodusDatabase, ExodusMesh}
)::Int32
  return Exodus.num_dimensions(mesh.mesh_obj.init)
end

function sidesets(mesh::FileMesh{<:ExodusDatabase, ExodusMesh})
  ids = convert(Vector{Int}, Exodus.read_ids(mesh.mesh_obj, SideSet))
  names = Exodus.read_names(mesh.mesh_obj, SideSet)
  ssets = read_sets(mesh.mesh_obj, SideSet)

  elems = Dict{String, Vector{Int}}()
  nodes = Dict{String, Vector{Int}}()
  sides = Dict{String, Vector{Int}}()
  side_nodes = Dict{String, Matrix{Int}}()

  blocks = nothing
  for (id, name, sset) in zip(ids, names, ssets)
    perm = sortperm(sset.elements)
    elems[name] = convert(Vector{Int}, sset.elements[perm])
    sides[name] = convert(Vector{Int}, sset.sides[perm])

    if length(sset.sides) == 0
      num_nodes_per_side
    else
      num_nodes_per_side = length(sset.side_nodes) ÷ length(sset.sides)
    end

    raw = convert(Vector{Int}, sset.side_nodes)
    if num_nodes_per_side == 7
      blocks = blocks === nothing ? read_sets(mesh.mesh_obj, Exodus.Block) : blocks
      _fill_seventh_tet_side_node!(raw, sset, blocks)
    end
    # A copy: the periodic boundary conditions resize this vector, and the
    # reshape below marks its argument as shared on Julia 1.10.
    nodes[name] = copy(raw)

    side_nodes[name] = reshape(
      reshape(raw, num_nodes_per_side, length(sset.sides))[:, perm],
      1, length(raw)
    )
  end

  names = Dict{Int, String}(zip(ids, names))
  return elems, names, nodes, sides, side_nodes
end

# The face node of each side of a 14- or 15-node tetrahedron: node 14 on side
# 1 (face 1-2-4), 12 on side 2 (2-3-4), 13 on side 3 (1-4-3), 11 on side 4
# (1-3-2).  This is the seventh column of the tetra_table in
# ex_get_side_set_node_list.c of the Exodus library.
const _TET15_SIDE_NODE = (14, 12, 13, 11)

# ex_get_side_set_node_list reports seven nodes per side of a 14- or 15-node
# tetrahedron but writes only six, so the seventh entry of every side holds
# whatever the buffer held before.  Fill it from the element connectivity.
# Side-set elements are internal element numbers, 1-based over the blocks in
# file order.
function _fill_seventh_tet_side_node!(raw::Vector{Int}, sset, blocks)
  counts = [size(b.conn, 2) for b in blocks]
  offsets = cumsum(vcat(0, counts))
  for (k, (e, s)) in enumerate(zip(sset.elements, sset.sides))
    b = searchsortedlast(offsets, e - 1) # block index with offsets[b] < e <= offsets[b + 1]
    b = min(b, length(blocks))
    conn = blocks[b].conn
    size(conn, 1) in (14, 15) ||
      error("Side set $(sset.id): seven nodes per side but element $e has " *
            "$(size(conn, 1)) nodes; only 14- and 15-node tetrahedra are handled.")
    1 <= s <= 4 || error("Side set $(sset.id): side $s of element $e is not a tetrahedron side.")
    raw[7 * k] = Int(conn[_TET15_SIDE_NODE[s], e - offsets[b]])
  end
  return raw
end

# additional optional interface
"""
$(TYPEDSIGNATURES)
"""
function nodal_coordinates(mesh::FileMesh{<:ExodusDatabase, ExodusMesh})
  coords = Exodus.read_coordinates(mesh.mesh_obj)
  return H1Field(coords)
end

function nodal_coordinates(type::Type{<:H1Field}, mesh::FileMesh{<:ExodusDatabase, ExodusMesh})
  coords = Exodus.read_coordinates(mesh.mesh_obj)
  return type(vec(coords))
end

"""
$(TYPEDSIGNATURES)
"""
function node_cmaps(mesh::FileMesh{<:ExodusDatabase, ExodusMesh}, rank)
  return Exodus.read_node_cmaps(rank, mesh.mesh_obj)
end

function node_id_map(
  mesh::FileMesh{<:ExodusDatabase, ExodusMesh}
)
  return convert(Vector{Int}, read_id_map(mesh.mesh_obj, NodeMap))
end

function num_nodes(
  mesh::FileMesh{<:ExodusDatabase, ExodusMesh}
)::Int32
  return Exodus.num_nodes(mesh.mesh_obj.init)
end

# TODO move below stuff to PostProcessor.jl since we'll always want to output
# to exodus.
# PostProcessor implementations
function PostProcessor(
  ::Type{<:ExodusMesh},
  file_name::String,
  vars...;
  extra_nodal_names::Vector{String} = String[]
)

  # scratch arrays for var names
  element_var_names = String[]
  nodal_var_names = String[]
  quadrature_var_names = String[]

  for var in vars
    if isa(var.fspace.coords, H1Field)
      append!(nodal_var_names, names(var))
    elseif isa(var.fspace.coords, L2Field)
      append!(element_var_names, names(var))
    # L2QuadratureField case below
    elseif isa(var.fspace.coords, NamedTuple)
      max_q_points = map(num_quadrature_points, values(var.fspace.ref_fes)) |> maximum
      temp_names = Symbol[]
      for name in names(var)
        for q in 1:max_q_points
          # push!(temp_names, Symbol(String(name) * "_$q"))
          push!(temp_names, "$(name)_$q")
        end
      end
      append!(quadrature_var_names, temp_names)
    else
      @assert false "Unsupported variable type currently $(typeof(var.fspace.coords))"
    end
  end

  # convert symbols to strings and append extra nodal names
  append!(nodal_var_names, extra_nodal_names)
  all_el_var_names = element_var_names
  append!(all_el_var_names, quadrature_var_names)

  # TODO need to add all the quadrature values labelled by block id

  exo = ExodusDatabase(file_name, "rw")

  # TODO setup element and quadrature fields as well
  if length(all_el_var_names) > 0
    write_names(exo, ElementVariable, all_el_var_names)
  end

  if length(nodal_var_names) > 0
    write_names(exo, NodalVariable, nodal_var_names)
  end
  Exodus.write_time(exo, 1, 0.0)
  
  return PostProcessor(
    file_name, exo,
    nodal_var_names, all_el_var_names, String[]
  )
end

function PostProcessor{O}(::Type{<:ExodusMesh}, mesh, output_file) where O <: ExodusDatabase
  copy_mesh(mesh, output_file, ExodusMesh, O)
  exo = O(output_file, "rw")
  # return PostProcessor{O}(output_file, exo, String[], String[], String[])
  return PostProcessor(output_file, exo, String[], String[], String[])
end

function write_field(
  pp::PostProcessor, time_index, block_name, field_name, field::Matrix{T}
) where T <: Number
  for q in axes(field, 1)
    var_name = "$(field_name)_$q"
    write_values(pp.field_output_db, ElementVariable, time_index, block_name, var_name, field[q, :])
  end
end

function write_field(
  pp::PostProcessor, time_index, block_name, field_name, field::Matrix{T}
) where T <: SymmetricTensor{2, 3, <:Number, 6}
  # SymmetricTensor{2,3} data order (column-major): xx, xy, xz, yy, yz, zz
  exts = ("(XX)", "(XY)", "(XZ)", "(YY)", "(YZ)", "(ZZ)")
  for (n, ext) in enumerate(exts)
    for q in axes(field, 1)
      var_name = "$(field_name)_$(ext)_$q"
      temp = map(x -> x.data[n], field[q, :])
      write_values(pp.field_output_db, ElementVariable, time_index, block_name, var_name, temp)
    end
  end
end

function write_field(
  pp::PostProcessor, time_index, block_name, field_name, field::Matrix{T}
) where T <: Tensor{2, 3, <:Number, 9}
  # Tensor{2,3} data order (column-major): xx, yx, zx, xy, yy, zy, xz, yz, zz
  exts = ("(XX)", "(YX)", "(ZX)", "(XY)", "(YY)", "(ZY)", "(XZ)", "(YZ)", "(ZZ)")
  for (n, ext) in enumerate(exts)
    for q in axes(field, 1)
      var_name = "$(field_name)_$(ext)_$q"
      temp = map(x -> x.data[n], field[q, :])
      write_values(pp.field_output_db, ElementVariable, time_index, block_name, var_name, temp)
    end
  end
end

function write_field(
  pp::PostProcessor, time_index, block_name, field_name, field::T
) where T <: AbstractArray{<:Number, 3}
  for q in axes(field, 2)
    for n in axes(field, 1)
      var_name = "$(field_name)_$n_$q"
      write_values(pp.field_output_db, ElementVariable, time_index, block_name, var_name, field[n, q, :])
    end
  end
end

function write_field(pp::PostProcessor, time_index::Int, field_names, field::H1Field)
  @assert length(field_names) == num_fields(field)
  for n in axes(field, 1)
    name = field_names[n]
    write_values(pp.field_output_db, NodalVariable, time_index, name, field[n, :])
  end
end

# DO SOMETHING ABOUT THIS ONE
function write_field(pp::PostProcessor, time_index::Int, field_names, field::NamedTuple)
  @assert length(field_names) == length(field)
  field_names = String.(field_names)
  for (block, val) in field
    for name in field_names
      # write_values(pp.field_output_db, ElementVariable, time_index, block, name, val)
    end
  end
end

function write_times(pp::PostProcessor, time_index::Int, time_val::Float64)
  Exodus.write_time(pp.field_output_db, time_index, time_val)
end
