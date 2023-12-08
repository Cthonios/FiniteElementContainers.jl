abstract type AbstractDofManager{NDof, N, Itype} end
num_nodes(::AbstractDofManager{NDof, N, Itype}) where {NDof, N, Itype}         = N
num_dofs_per_node(::AbstractDofManager{NDof, N, Itype}) where {NDof, N, Itype} = NDof

function dof_ids(dof::Dof) where Dof <: AbstractDofManager
  NDofs, N = num_dofs_per_node(dof), num_nodes(dof)
  ids = reshape(1:N * NDofs, NDofs, N)
  return ids
end

########################################3

struct DofManager{NDof, N, Itype, B <: AbstractArray, V <: AbstractArray{Itype, 1}} <: AbstractDofManager{NDof, N, Itype}
  is_unknown::B
  unknown_indices::V
end

# default is to set up everything as free dof
function DofManager(mesh::Mesh, n_dofs::Int)
  n_nodes = num_nodes(mesh.coords)

  is_unknown      = BitArray(1 for _ = 1:n_dofs, _ = 1:n_nodes)
  ids             = reshape(1:length(is_unknown), n_dofs, n_nodes)
  unknown_indices = ids[is_unknown]

  return DofManager{n_dofs, n_nodes, Int64, typeof(is_unknown), typeof(unknown_indices)}(
    is_unknown, unknown_indices
  )
end

function create_field(d::DofManager, name::Symbol)
  NDof, N = num_dofs_per_node(d), num_nodes(d)
  return zeros(NodalField{NDof, N, Matrix, Float64}, name)
end

function create_unknowns(dof::DofManager)
  N = length(dof.unknown_indices)
  return zeros(N)
end

# this method simply frees all dofs
function update_unknown_ids!(dof::DofManager)
  dof.is_unknown .= 1
end

# this methods will set a given set of nodes for a given dof fixed

# function update_unknown_ids!(dof_manager::DofManager, nodes::V, dof::Int) where V <: AbstractArray{<:Integer, 1}
#   for node in nodes
#     dof_manager.is_unknown[dof, node] = 0
#   end

#   # TODO below line is the only source of allocations here
#   new_unknown_indices = dof_ids(dof)
#   # new_unknown_indices = dof_ids(dof)[dof.is_unknown]
# 	# resize!(dof.unknown_indices, length(new_unknown_indices))
# 	# dof.unknown_indices .= new_unknown_indices
# end

# this does it for a list of nodesets
function update_unknown_ids!(dof_manager::DofManager, nsets::V, dof::Int) where V <: AbstractArray{<:AbstractArray{<:Integer, 1}}
  for nset in nsets
    for node in nset
      dof_manager.is_unknown[dof, node] = 0
    end
  end

  resize!(dof_manager.unknown_indices, sum(dof_manager.is_unknown))

  # this has two allocations but twice as fast as below
  dof_manager.unknown_indices .= dof_ids(dof_manager)[dof_manager.is_unknown]

  # below has zero allocations but is twice as slow as above
  # n = 1
  # for j in axes(dof_manager.is_unknown, 2)
  #   for i in axes(dof_manager.is_unknown, 1)
  #     if dof_manager.is_unknown[i, j]
  #       dof_manager.unknown_indices[n] = dof_manager.is_unknown[i, j]
  #       n = n + 1
  #     end
  #   end
  # end
end

function dof_connectivity(dof::DofManager, conn::Connectivity)
  NN, E    = num_nodes_per_element(conn), num_elements(conn)
  NDof     = num_dofs_per_node(dof)
  ids      = dof_ids(dof)
  dof_conn = @views reshape(ids[:, conn], NDof * NN, E)
  return dof_conn
end

function dof_connectivity(dof::DofManager, conn::Connectivity, e::Int)
  NN       = num_nodes_per_element(conn)
  NDof     = num_dofs_per_node(dof)
  ids      = dof_ids(dof)
  # dof_conn = @views reshape(ids[:, connectivity(conn, e)], NDof * NN)
  # return dof_conn
  # dof_conn = @views reinterpret(SVector{NN * NDof, eltype(conn)}, vec(ids[:, connectivity(conn, e)]))
  dof_conn = @views vec(ids[:, connectivity(conn, e)])
end

dof_connectivity(dof::DofManager, fspace::F) where F <: AbstractFunctionSpace = 
dof_connectivity(dof, connectivity(fspace))

dof_connectivity(dof::DofManager, fspace::F, e::Int) where F <: AbstractFunctionSpace =
dof_connectivity(dof, connectivity(fspace), e)

function update_fields!(U::NodalField, d::DofManager, Uu::V) where V <: AbstractVector
  @assert length(Uu) == sum(d.is_unknown)
  U[d.is_unknown] = Uu
end 