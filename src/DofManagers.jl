abstract type AbstractDofManager{T, N, NDofs, NNodes, Bools} <: AbstractArray{T, N} end
num_nodes(::AbstractDofManager{T, N, NDofs, NNodes, Bools}) where {T, N, NDofs, NNodes, Bools}         = NNodes
num_dofs_per_node(::AbstractDofManager{T, N, NDofs, NNodes, Bools}) where {T, N, NDofs, NNodes, Bools} = NDofs
ids_type(::AbstractDofManager{T, N, NDofs, NNodes, Bools}) where {T, N, NDofs, NNodes, Bools}          = Bools

function dof_ids(dof::Dof) where Dof <: AbstractDofManager
  @assert ids_type(dof) <: AbstractVecOrMat
  NDofs, N = num_dofs_per_node(dof), num_nodes(dof)
  ids      = 1:N * NDofs
  if ids_type(dof) <: AbstractMatrix
    ids = reshape(ids, NDofs, N)
  end
end

struct DofManager{
  N, NDofs, NNodes,
  Bools <: BitArray{N},
  IDs   <: AbstractArray{<:Integer, 1}
} <: AbstractDofManager{Bool, N, NDofs, NNodes, Bools}

  is_unknown::Bools
  unknown_indices::IDs
end

Base.IndexStyle(::Type{<:DofManager})          = IndexLinear()
Base.size(dof::DofManager)                     = (num_dofs_per_node(dof), num_nodes(dof))
Base.getindex(dof::DofManager, index::Int)     = getindex(dof.is_unknown, index)
Base.setindex!(dof::DofManager, v, index::Int) = setindex!(dof.is_unknown, v, index)
Base.axes(dof::DofManager)                     = axes(dof.is_unknown)

function Base.getindex(dof::DofManager, d::Int, n::Int)
  if ids_type(dof) <: BitVector
    @assert d > 0
    @assert n > 0
    @assert d <= num_dofs_per_node(dof)
    @assert n <= num_nodes(dof)
    return getindex(dof.is_unknown, (n - 1) * num_dofs_per_node(dof) + d)
  else
    return getindex(dof.is_unknown, d, n)
  end
end 

# Base.setindex!(dof::DofManager, v, index::Int) = setindex!(dof.is_unknown, v, index)
function Base.setindex!(dof::DofManager, v, d::Int, n::Int)
  if ids_type(dof) <: BitVector
    @assert d > 0
    @assert n > 0
    @assert d <= num_dofs_per_node(dof)
    @assert n <= num_nodes(dof)
    setindex!(dof.is_unknown, v, (n - 1) * num_dofs_per_node(dof) + d)
  else
    setindex!(dof.is_unknown, v, d, n)
  end
end

Base.axes(dof::DofManager, ::Val{1}) = Base.OneTo(num_dofs_per_node(dof))
Base.axes(dof::DofManager, ::Val{2}) = Base.OneTo(num_nodes(dof))

function Base.axes(dof::DofManager, n::Int)
  @assert n > 0
  @assert n <= 2
  return axes(dof, Val(n))
end

function DofManager{NDofs, NNodes, Vector}() where {NDofs, NNodes}
  is_unknown      = BitVector(1 for _ = 1:NDofs * NNodes)
  ids             = 1:length(is_unknown)
  unknown_indices = ids[is_unknown]
  return DofManager{1, NDofs, NNodes, typeof(is_unknown), typeof(unknown_indices)}(
    is_unknown, unknown_indices
  )
end

function DofManager{NDofs, NNodes, Matrix}() where {NDofs, NNodes}
  is_unknown      = BitArray(1 for _ = 1:NDofs, _ = 1:NNodes)
  ids             = reshape(1:length(is_unknown), NDofs, NNodes)
  unknown_indices = ids[is_unknown]
  return DofManager{2, NDofs, NNodes, typeof(is_unknown), typeof(unknown_indices)}(
    is_unknown, unknown_indices
  )
end

function create_field(dof::DofManager, name::Symbol) 
  NDof, N = num_dofs_per_node(dof), num_nodes(dof)
  if ndims(dof.is_unknown) > 1
    type = Matrix
  else
    type = Vector
  end
  return zeros(NodalField{NDof, N, type, Float64}, name)
end

create_unknowns(dof::DofManager) = zeros(length(dof.unknown_indices))

# this method simply frees all dofs
function update_unknown_ids!(dof::DofManager)
  dof.is_unknown .= 1
end

function update_unknown_ids!(
  dof_manager::DofManager, 
  nodes::Nodes, dof::Int
) where Nodes <: AbstractArray{<:Integer, 1}
        
  for node in nodes
    dof_manager[dof, node] = 0
  end
end

function update_unknown_ids!(
  dof_manager::DofManager,
  nsets::Nodes, dofs::Dofs
) where {Nodes <: AbstractArray{<:AbstractArray{<:Integer, 1}, 1},
         Dofs  <: AbstractArray{<:Integer}}

  for (n, nset) in enumerate(nsets)
    @assert dofs[n] >= num_dofs_per_node(dof_manager)
    update_unknown_ids!(dof_manager, nset, dofs[n])
  end
  resize!(dof_manager.unknown_indices, sum(dof_manager.is_unknown))
  dof_manager.unknown_indices .= dof_ids(dof_manager)[dof_manager.is_unknown]
end

function update_fields!(U::NF, dof::DofManager, Uu::V) where {NF <: NodalField, V <: AbstractVector{<:Number}}
  @assert length(Uu) == sum(dof.is_unknown)
  U[dof.is_unknown] = Uu
end