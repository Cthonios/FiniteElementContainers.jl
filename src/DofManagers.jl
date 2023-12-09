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

function DofManager{NDofs, NNodes, Vector}() where {NDofs, NNodes}
  is_unknown      = BitVector(1 for _ = 1:NDofs * NNodes)
  ids             = 1:length(is_unknown)
  unknown_indices = ids[is_unknown]

  # return DofManager{n_dofs, n_nodes, Int64, typeof(is_unknown), typeof(unknown_indices)}(
  return DofManager{NDofs, NNodes, Int64, typeof(is_unknown), typeof(unknown_indices)}(
    is_unknown, unknown_indices
  )
end

# default is to set up everything as free dof
function DofManager{NDofs}(mesh::Mesh, ::Type{Vector}) where NDofs
  n_nodes = num_nodes(mesh.coords)

  # is_unknown      = BitArray(1 for _ = 1:n_dofs, _ = 1:n_nodes)
  # ids             = reshape(1:length(is_unknown), n_dofs, n_nodes)
  is_unknown      = BitVector(1 for _ = 1:NDofs * n_nodes)
  ids             = 1:length(is_unknown)
  unknown_indices = ids[is_unknown]

  # return DofManager{n_dofs, n_nodes, Int64, typeof(is_unknown), typeof(unknown_indices)}(
  return DofManager{NDofs, n_nodes, Int64, typeof(is_unknown), typeof(unknown_indices)}(
    is_unknown, unknown_indices
  )
end

function DofManager{NDofs, NNodes, Matrix}() where {NDofs, NNodes}
  is_unknown      = BitArray(1 for _ = 1:NDofs, _ = 1:NNodes)
  ids             = reshape(1:length(is_unknown), NDofs, NNodes)
  unknown_indices = ids[is_unknown]

  return DofManager{NDofs, NNodes, Int64, typeof(is_unknown), typeof(unknown_indices)}(
    is_unknown, unknown_indices
  )
end

function DofManager{NDofs}(mesh::Mesh, ::Type{Matrix}) where NDofs
  n_nodes = num_nodes(mesh.coords)

  # is_unknown      = BitArray(1 for _ = 1:n_dofs, _ = 1:n_nodes)
  # ids             = reshape(1:length(is_unknown), n_dofs, n_nodes)
  is_unknown      = BitArray(1 for _ = 1:NDofs, _ = 1:n_nodes)
  ids             = reshape(1:length(is_unknown), NDofs, n_nodes)
  unknown_indices = ids[is_unknown]

  # return DofManager{n_dofs, n_nodes, Int64, typeof(is_unknown), typeof(unknown_indices)}(
  return DofManager{NDofs, n_nodes, Int64, typeof(is_unknown), typeof(unknown_indices)}(
    is_unknown, unknown_indices
  )
end

function create_field(d::DofManager, name::Symbol, ::Type{T}) where T <: AbstractArray
  NDof, N = num_dofs_per_node(d), num_nodes(d)
  return zeros(NodalField{NDof, N, T, Float64}, name)
end

function create_unknowns(dof::DofManager)
  N = length(dof.unknown_indices)
  return zeros(N)
end

# this method simply frees all dofs
function update_unknown_ids!(dof::DofManager)
  dof.is_unknown .= 1
end

# this does it for a list of nodesets
# below is allocation free
function update_unknown_ids!(
  dof_manager::DofManager{NDof, N, Itype, B, V1}, nsets::V2, dof::Int
) where {NDof, N, Itype, B <: BitVector, V1 <: AbstractArray, V2 <: AbstractArray{<:AbstractArray{<:Integer, 1}}}
  for nset in nsets
    for node in nset
      k = (node - 1) * NDof + dof
      dof_manager.is_unknown[k] = 0
    end
  end
  resize!(dof_manager.unknown_indices, sum(dof_manager.is_unknown))
  dof_manager.unknown_indices .= dof_ids(dof_manager)[dof_manager.is_unknown]
end

function update_unknown_ids!(
  dof_manager::DofManager{NDof, N, Itype, B, V1}, nsets::V2, dof::Int
) where {NDof, N, Itype, B <: BitMatrix, V1 <: AbstractArray, V2 <: AbstractArray{<:AbstractArray{<:Integer, 1}}}
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

# function dof_connectivity(dof::DofManager, conn::Connectivity)
#   NN, E    = num_nodes_per_element(conn), num_elements(conn)
#   NDof     = num_dofs_per_node(dof)
#   ids      = dof_ids(dof)
#   dof_conn = @views reshape(ids[:, conn], NDof * NN, E)
#   return dof_conn
# end

# function dof_connectivity(dof::DofManager, conn::Connectivity, e::Int)
#   NN       = num_nodes_per_element(conn)
#   NDof     = num_dofs_per_node(dof)
#   ids      = dof_ids(dof)
#   # dof_conn = @views reshape(ids[:, connectivity(conn, e)], NDof * NN)
#   # return dof_conn
#   # dof_conn = @views reinterpret(SVector{NN * NDof, eltype(conn)}, vec(ids[:, connectivity(conn, e)]))
#   dof_conn = @views vec(ids[:, connectivity(conn, e)])
# end

# dof_connectivity(dof::DofManager, fspace::F) where F <: AbstractFunctionSpace = 
# dof_connectivity(dof, connectivity(fspace))

# dof_connectivity(dof::DofManager, fspace::F, e::Int) where F <: AbstractFunctionSpace =
# dof_connectivity(dof, connectivity(fspace), e)

function update_fields!(
  U::NodalField{T, N, NF, NN, Name, V1}, d::DofManager, Uu::V2
) where {T, N, NF, NN, Name, V1 <: AbstractArray{T, N}, V2 <: AbstractVector}
  @assert length(Uu) == sum(d.is_unknown)
  U[d.is_unknown] = Uu
end 