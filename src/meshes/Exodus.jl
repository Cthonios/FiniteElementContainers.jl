"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
function FileMesh(::ExodusMesh, file_name::String)
  exo = ExodusDatabase(file_name, "r")
  return FileMesh{typeof(exo)}(file_name, exo)
end

function _mesh_file_type(::Type{ExodusMesh})
  return ExodusDatabase
end

function num_dimensions(
  mesh::FileMesh{ExodusDatabase{M, I, B, F}}
)::Int32 where {M, I, B, F}
  return Exodus.num_dimensions(mesh.mesh_obj.init)
end

function num_nodes(
  mesh::FileMesh{<:ExodusDatabase}
)::Int32
  return Exodus.num_nodes(mesh.mesh_obj.init)
end

"""
$(TYPEDSIGNATURES)
"""
function element_block_id_map(mesh::FileMesh{<:ExodusDatabase}, id)
  return convert.(Int64, Exodus.read_block_id_map(mesh.mesh_obj, id))
end

"""
$(TYPEDSIGNATURES)
"""
function element_block_ids(mesh::FileMesh{<:ExodusDatabase})
  return Exodus.read_ids(mesh.mesh_obj, Block)
end

"""
$(TYPEDSIGNATURES)
"""
function element_block_names(mesh::FileMesh{<:ExodusDatabase})
  return Exodus.read_names(mesh.mesh_obj, Block)
end

"""
$(TYPEDSIGNATURES)
"""
function node_cmaps(mesh::FileMesh{<:ExodusDatabase}, rank)
  return Exodus.read_node_cmaps(rank, mesh.mesh_obj)
end
"""
$(TYPEDSIGNATURES)
"""
function nodeset_ids(mesh::FileMesh{<:ExodusDatabase})
  return Exodus.read_ids(mesh.mesh_obj, NodeSet)
end

"""
$(TYPEDSIGNATURES)
"""
function nodeset_names(mesh::FileMesh{<:ExodusDatabase})
  return Exodus.read_names(mesh.mesh_obj, NodeSet)
end

"""
$(TYPEDSIGNATURES)
"""
function sideset_ids(mesh::FileMesh{<:ExodusDatabase})
  return Exodus.read_ids(mesh.mesh_obj, SideSet)
end

"""
$(TYPEDSIGNATURES)
"""
function sideset_names(mesh::FileMesh{<:ExodusDatabase})
  return Exodus.read_names(mesh.mesh_obj, SideSet)
end

"""
$(TYPEDSIGNATURES)
"""
function coordinates(mesh::FileMesh{ExodusDatabase{M, I, B, F}})::Matrix{F} where {M, I, B, F} 
  coords = Exodus.read_coordinates(mesh.mesh_obj)
  return coords
end

function element_connectivity(
  mesh::FileMesh{<:ExodusDatabase},
  id::Integer
) 

  block = read_block(mesh.mesh_obj, id)
  return convert.(Int64, block.conn)
end

function element_connectivity(
  mesh::FileMesh{<:ExodusDatabase},
  name::String
) 

  block = read_block(mesh.mesh_obj, name)
  return block.conn
end


function element_type(
  mesh::FileMesh{<:ExodusDatabase},
  id::Integer
) 
  block = read_block(mesh.mesh_obj, id)
  return block.elem_type
end

function element_type(
  mesh::FileMesh{<:ExodusDatabase},
  name::String
) 
  block = read_block(mesh.mesh_obj, name)
  return block.elem_type
end

function copy_mesh(file_1::String, file_2::String)
  Exodus.copy_mesh(file_1, file_2)
end

function node_id_map(
  mesh::FileMesh{<:ExodusDatabase}
)
  return read_id_map(mesh.mesh_obj, NodeMap)
end

function nodeset(
  mesh::FileMesh{<:ExodusDatabase},
  id::Integer
) 
  nset = read_set(mesh.mesh_obj, NodeSet, id)
  return sort!(convert.(Int64, nset.nodes))
end

function nodesets(
  mesh::FileMesh{<:ExodusDatabase},
  ids
) 
  return nodeset.((mesh,), ids)
end

function sideset(
  mesh::FileMesh{<:ExodusDatabase},
  id::Integer
) 
  sset = read_set(mesh.mesh_obj, SideSet, id)
  elems = convert.(Int64, sset.elements)
  sides = convert.(Int64, sset.sides)
  nodes = convert.(Int64, Exodus.read_side_set_node_list(mesh.mesh_obj, id)[2])
  side_nodes = convert.(Int64, sset.side_nodes)

  # re-arrange some of these
  unique!(sort!(nodes))
  
  # side_nodes = reshape(side_nodes, 1, length(side_nodes))
  perm = sortperm(elems)

  if length(sides) == 0
    num_nodes_per_side = 0
  else
    num_nodes_per_side = length(side_nodes) ÷ length(sides)
  end
  side_nodes = reshape(side_nodes, num_nodes_per_side, length(sides))[:, perm]
  side_nodes = reshape(side_nodes, 1, length(side_nodes))

  return elems[perm], nodes, sides[perm], side_nodes
end

function sidesets(
  mesh::FileMesh{<:ExodusDatabase},
  ids
) 
  return sideset.((mesh,), ids)
end


# TODO move below stuff to PostProcessor.jl since we'll always want to output
# to exodus.
# PostProcessor implementations
function PostProcessor(
  ::Type{<:ExodusMesh},
  file_name::String, 
  vars...
)

  # scratch arrays for var names
  element_var_names = Symbol[]
  nodal_var_names = Symbol[]
  quadrature_var_names = Symbol[]

  for var in vars
    if isa(var.fspace.coords, H1Field)
      append!(nodal_var_names, names(var))
    elseif isa(var.fspace.coords, L2ElementField)
      append!(element_var_names, names(var))
    # L2QuadratureField case below
    elseif isa(var.fspace.coords, NamedTuple)
      max_q_points = mapreduce(num_quadrature_points, maximum, values(var.fspace.ref_fes))
      temp_names = Symbol[]
      for name in names(var)
        for q in 1:max_q_points
          push!(temp_names, Symbol(String(name) * "_$q"))
        end
      end
      append!(quadrature_var_names, temp_names)
    else
      @assert false "Unsupported variable type currently $(typeof(var.fspace.coords))"
    end
  end

  # convert symbols to strings
  nodal_var_names = String.(nodal_var_names)
  # all_el_var_names = String.(vcat(element_var_names, quadrature_var_names))
  all_el_var_names = element_var_names
  append!(all_el_var_names, quadrature_var_names)
  all_el_var_names = String.(all_el_var_names)

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
  
  return PostProcessor(exo)
end

function write_field(pp::PostProcessor, time_index, block_name, field_name, field::Matrix{<:Number})
  for q in axes(field, 1)
    var_name = String(field_name) * "_$q"
    write_values(pp.field_output_db, ElementVariable, time_index, block_name, var_name, field[q, :])
  end
end

function write_field(pp::PostProcessor, time_index, block_name, field_name, field::Matrix{<:SymmetricTensor{2, 3, <:Number, 6}})
  exts = ("xx", "yy", "zz", "yz", "xz", "xy")
  for (n, ext) in enumerate(exts)
    for q in axes(field, 1)
      var_name = String(field_name) * "_$(ext)_$q"
      temp = map(x -> x.data[n], field[q, :])
      write_values(pp.field_output_db, ElementVariable, time_index, block_name, var_name, temp)
    end
  end
end

function write_field(pp::PostProcessor, time_index, block_name, field_name, field::Matrix{<:Tensor{2, 3, <:Number, 9}})
  exts = ("xx", "yy", "zz", "yz", "xz", "xy", "zy", "zx", "yx")
  for (n, ext) in enumerate(exts)
    for q in axes(field, 1)
      var_name = String(field_name) * "_$(ext)_$q"
      temp = map(x -> x.data[n], field[q, :])
      write_values(pp.field_output_db, ElementVariable, time_index, block_name, var_name, temp)
    end
  end
end

function write_field(pp::PostProcessor, time_index::Int, field_names, field::H1Field)
  @assert length(field_names) == num_fields(field)
  for n in axes(field, 1)
    name = String(field_names[n])
    write_values(pp.field_output_db, NodalVariable, time_index, name, field[n, :])
  end
end

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
