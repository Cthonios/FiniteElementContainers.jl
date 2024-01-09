abstract type DofManager{T, N, ND, NN, Bools} <: NodalField{T, N, ND, NN, Bools} end
num_dofs_per_node(dof::DofManager) = num_fields(dof)

function create_fields(
  ::DofManager{T, N, ND, NN, B}, 
  # float_type::Type{<:Number} = Float64
  type::Type = Float64
) where {T, N, ND, NN, B <: BitArray{1}}

  vals = zeros(type, ND, NN)
  return NodalField{ND, NN, Vector}(vals)
end

function create_fields(
  ::DofManager{T, N, ND, NN, B}, 
  # float_type::Type{<:Number} = Float64
  type::Type = Float64
) where {T, N, ND, NN, B <: BitArray{2}}

  vals = zeros(type, ND, NN)
  return NodalField{ND, NN, Matrix}(vals)
end

function create_unknowns(
  dof::DofManager{T, N, ND, NN, Bools}, 
  type = Float64
) where {T, N, ND, NN, Bools}
  return zeros(type, length(dof.unknown_indices))
end

function dof_ids(::DofManager{T, N, ND, NN, Bools}) where {T, N, ND, NN, Bools <: AbstractArray{Bool, 1}}
  ids = 1:ND * NN
  return ids
end

function dof_ids(::DofManager{T, N, ND, NN, Bools}) where {T, N, ND, NN, Bools <: AbstractArray{Bool, 2}}
  ids = reshape(1:ND * NN, ND, NN)
  return ids
end

function update_fields!(U::NodalField, dof::DofManager, Uu::V) where V <: AbstractArray{<:Number, 1}
  # @assert length(Uu) == sum(dof.is_unknown)
  # U[dof.is_unknown] = Uu
  @assert length(Uu) == length(dof.unknown_indices)
  U[dof.unknown_indices] = Uu
end

# function update_fields!(
#   U::NodalField{T, N, ND, NN, Name, V1}, d::DofManager, Uu::V2
# ) where {T, N, NF, NN, Name, V1 <: AbstractArray{T, N}, V2 <: AbstractVector}
#   @assert length(Uu) == sum(d.is_unknown)
#   U[d.is_unknown] = Uu
# end 

function update_unknown_ids!(dof::DofManager)
  dof.is_unknown .= 1
  resize!(dof.unknown_indices, sum(dof.is_unknown))
  ids = dof_ids(dof)
  dof.unknown_indices .= ids[dof.is_unknown]
end

function update_unknown_ids!(
  dof_manager::DofManager, nodes::Nodes, dof::Int
) where Nodes #<: AbstractArray{<:Integer, 1}
  
  for node in nodes
    dof_manager[dof, node] = 0
  end
end

function update_unknown_ids!(
  dof_manager::DofManager, nsets::NSets, dofs::Dofs
) where {NSets, Dofs} 

#where {NSets <: AbstractArray{<:AbstractArray{<:Integer, 1}, 1}, Dofs <: AbstractArray{<:Integer, 1}}

  @assert length(nsets) == length(dofs)
  for (n, nset) in enumerate(nsets)
    @assert dofs[n] > 0 && dofs[n] <= num_dofs_per_node(dof_manager)
    update_unknown_ids!(dof_manager, nset, dofs[n])
  end
  resize!(dof_manager.unknown_indices, sum(dof_manager.is_unknown))
  ids = dof_ids(dof_manager)
  dof_manager.unknown_indices .= ids[dof_manager.is_unknown]
end

##########################################################################

# struct SimpleDofManager{
#   T, N, ND, NN, Bools <: BitArray{2}, Indices <: AbstractArray{<:Integer, 1}
# } <: DofManager{T, N, ND, NN, Bools}
#   is_unknown::Bools
#   unknown_indices::Indices
# end

struct SimpleDofManager{
  T, N, ND, NN, Bools <: AbstractArray{Bool, 2}, Indices <: AbstractArray{<:Integer, 1}
} <: DofManager{T, N, ND, NN, Bools}
  is_unknown::Bools
  unknown_indices::Indices
end

Base.IndexStyle(::Type{<:SimpleDofManager}) = IndexLinear()

Base.axes(field::SimpleDofManager) = axes(field.is_unknown)
Base.getindex(field::SimpleDofManager, n::Int) = getindex(field.is_unknown, n)
Base.setindex!(field::SimpleDofManager, v, n::Int) = setindex!(field.is_unknown, v, n)
Base.size(::SimpleDofManager{T, N, ND, NN, V}) where {T, N, ND, NN, V} = (ND, NN)


function SimpleDofManager{ND, NN}() where {ND, NN}
  is_unknown      = BitArray(1 for _ = 1:ND, _ = 1:NN)
  ids             = reshape(1:ND * NN, ND, NN)
  unknown_indices = ids[is_unknown]
  return SimpleDofManager{Bool, 2, ND, NN, typeof(is_unknown), typeof(unknown_indices)}(is_unknown, unknown_indices)
end

##########################################################################

struct VectorizedDofManager{
  T, N, ND, NN, Bools <: AbstractArray{Bool, 1}, Indices <: AbstractArray{<:Integer, 1}
} <: DofManager{T, N, ND, NN, Bools}
  is_unknown::Bools
  unknown_indices::Indices
end

Base.IndexStyle(::Type{<:VectorizedDofManager}) = IndexLinear()
Base.axes(dof::VectorizedDofManager) = (Base.OneTo(num_dofs_per_node(dof)), Base.OneTo(num_nodes(dof)))
Base.getindex(dof::VectorizedDofManager, n::Int) = getindex(dof.is_unknown, n)
function Base.getindex(dof::VectorizedDofManager, d::Int, n::Int) 
  @assert d > 0 && d <= num_dofs_per_node(dof)
  @assert n > 0 && n <= num_nodes(dof)
  getindex(dof.is_unknown, (n - 1) * num_dofs_per_node(dof) + d)
end
Base.setindex!(dof::VectorizedDofManager, v, n::Int) = setindex!(dof.is_unknown, v, n)
function Base.setindex!(dof::VectorizedDofManager, v, d::Int, n::Int)
  @assert d > 0 && d <= num_dofs_per_node(dof)
  @assert n > 0 && n <= num_nodes(dof)
  setindex!(dof.is_unknown, v, (n - 1) * num_dofs_per_node(dof) + d)
end
Base.size(::VectorizedDofManager{T, N, ND, NN, V}) where {T, N, ND, NN, V} = (ND, NN)

function VectorizedDofManager{ND, NN}() where {ND, NN}
  is_unknown      = BitVector(1 for _ = 1:ND * NN)
  ids             = 1:ND * NN
  unknown_indices = ids[is_unknown]
  return VectorizedDofManager{Bool, 2, ND, NN, typeof(is_unknown), typeof(unknown_indices)}(is_unknown, unknown_indices)
end

##########################################################################

DofManager{ND, NN, Matrix}() where {ND, NN} = SimpleDofManager{ND, NN}()
DofManager{ND, NN, Vector}() where {ND, NN} = VectorizedDofManager{ND, NN}()

# abstract type AbstractDofManager{T, N, NDofs, NNodes, Bools} <: AbstractArray{T, N} end
# # abstract type AbstractDofManager{T, N, NDofs, NNodes, Bools} end
# num_nodes(::AbstractDofManager{T, N, NDofs, NNodes, Bools}) where {T, N, NDofs, NNodes, Bools}         = NNodes
# num_dofs_per_node(::AbstractDofManager{T, N, NDofs, NNodes, Bools}) where {T, N, NDofs, NNodes, Bools} = NDofs
# ids_type(::AbstractDofManager{T, N, NDofs, NNodes, Bools}) where {T, N, NDofs, NNodes, Bools}          = Bools

# function dof_ids(dof::Dof) where Dof <: AbstractDofManager
#   @assert ids_type(dof) <: AbstractVecOrMat
#   NDofs, N = num_dofs_per_node(dof), num_nodes(dof)
#   ids      = 1:N * NDofs
#   if ids_type(dof) <: AbstractMatrix
#     ids = reshape(ids, NDofs, N)
#   end
#   return ids
# end

# ###############################################################

# struct DofManager{
#   N, NDofs, NNodes,
#   Bools <: BitArray{N},
#   IDs   <: AbstractArray{<:Integer, 1}
# } <: AbstractDofManager{Bool, N, NDofs, NNodes, Bools}

#   is_unknown::Bools
#   unknown_indices::IDs
# end

# Base.IndexStyle(::Type{<:DofManager})          = IndexLinear()
# Base.size(dof::DofManager)                     = (num_dofs_per_node(dof), num_nodes(dof))
# Base.getindex(dof::DofManager, index::Int)     = getindex(dof.is_unknown, index)
# Base.setindex!(dof::DofManager, v, index::Int) = setindex!(dof.is_unknown, v, index)
# # Base.axes(dof::D) where D <: DofManager        = axes(dof.is_unknown)
# Base.axes(dof::DofManager)                     = (axes(dof, 1), axes(dof, 2))

# # Base.size(::DofManager{N, NDofs, NNodes, Bools, IDs}) where {
# #   N, NDofs, NNodes, Bools <: BitVector, IDs
# # } = (NDofs, NNodes)

# # Base.size(::DofManager{N, NDofs, NNodes, Bools, IDs}) where {
# #   N, NDofs, NNodes, Bools <: BitMatrix, IDs
# # } = (NDofs, NNodes)


# function Base.getindex(dof::DofManager, d::Int, n::Int)
#   if ids_type(dof) <: BitVector
#     @assert d > 0
#     @assert n > 0
#     @assert d <= num_dofs_per_node(dof)
#     @assert n <= num_nodes(dof)
#     return getindex(dof.is_unknown, (n - 1) * num_dofs_per_node(dof) + d)
#   else
#     return getindex(dof.is_unknown, d, n)
#   end
# end 

# function Base.setindex!(dof::DofManager, v, d::Int, n::Int)
#   if ids_type(dof) <: BitVector
#     @assert d > 0
#     @assert n > 0
#     @assert d <= num_dofs_per_node(dof)
#     @assert n <= num_nodes(dof)
#     setindex!(dof.is_unknown, v, (n - 1) * num_dofs_per_node(dof) + d)
#   else
#     setindex!(dof.is_unknown, v, d, n)
#   end
# end


# # Base.axes(::DofManager{N, NDofs, NNodes, Bools, IDs}, ::Val{1}) where {
# #   N, NDofs, NNodes, Bools <: BitVector, IDs
# # } = Base.OneTo(NDofs)

# # Base.axes(::DofManager{N, NDofs, NNodes, Bools, IDs}, ::Val{2}) where {
# #   N, NDofs, NNodes, Bools <: BitVector, IDs
# # } = Base.OneTo(NNodes)

# Base.axes(dof::DofManager, ::Val{1}) = Base.OneTo(num_dofs_per_node(dof))
# Base.axes(dof::DofManager, ::Val{2}) = Base.OneTo(num_nodes(dof))

# function Base.axes(u::DofManager{N, NDofs, NNodes, Bools, IDs}, n::Int) where {
#   N, NDofs, NNodes, Bools <: BitVector, IDs
# } 
#   axes(u, Val(n))
# end

# # function Base.axes(u::DofManager{N, NDofs, NNodes, Bools, IDs}) where {
# #   N, NDofs, NNodes, Bools <: BitVector, IDs
# # } 
# #   return (axes(u, Val(1)), axes(u, Val(2)))
# # end

# # function Base.axes(u::NodalField{T, N, NDofs, NNodes, Bools}) where {
# #   T, N, NDofs, NNodes, Bools <: BitVector
# # } 
# #   axes(u, Val(n))
# # end




# # Base.axes(dof::DofManager, ::Val{1}) = Base.OneTo(num_dofs_per_node(dof))
# # Base.axes(dof::DofManager, ::Val{2}) = Base.OneTo(num_nodes(dof))

# # function Base.axes(dof::DofManager, n::Int)
# #   @assert n > 0
# #   @assert n <= 2
# #   return axes(dof, Val(n))
# # end

# function DofManager{NDofs, NNodes, Vector}() where {NDofs, NNodes}
#   is_unknown      = BitVector(1 for _ = 1:NDofs * NNodes)
#   ids             = 1:length(is_unknown)
#   unknown_indices = ids[is_unknown]
#   return DofManager{1, NDofs, NNodes, typeof(is_unknown), typeof(unknown_indices)}(
#     is_unknown, unknown_indices
#   )
# end

# function DofManager{NDofs, NNodes, Matrix}() where {NDofs, NNodes}
#   is_unknown      = BitArray(1 for _ = 1:NDofs, _ = 1:NNodes)
#   ids             = reshape(1:length(is_unknown), NDofs, NNodes)
#   unknown_indices = ids[is_unknown]
#   return DofManager{2, NDofs, NNodes, typeof(is_unknown), typeof(unknown_indices)}(
#     is_unknown, unknown_indices
#   )
# end

# function create_field(dof::DofManager, name::Symbol) 
#   NDof, N = num_dofs_per_node(dof), num_nodes(dof)
#   if ndims(dof.is_unknown) > 1
#     type = Matrix
#   else
#     type = Vector
#   end
#   return zeros(NodalField{NDof, N, type, Float64}, name)
# end

# create_unknowns(dof::DofManager) = zeros(length(dof.unknown_indices))

# # this method simply frees all dofs
# function update_unknown_ids!(dof::DofManager)
#   dof.is_unknown .= 1
#   resize!(dof.unknown_indices, sum(dof.is_unknown))
#   dof.unknown_indices .= dof_ids(dof)[dof.is_unknown]
# end

# function update_unknown_ids!(
#   dof_manager::DofManager, 
#   nodes::Nodes, dof::Int
# ) where Nodes <: AbstractArray{<:Integer, 1}
        
#   for node in nodes
#     dof_manager[dof, node] = 0
#   end
# end

# function update_unknown_ids!(
#   dof_manager::DofManager,
#   nsets::Nodes, dofs::Dofs
# ) where {Nodes <: AbstractArray{<:AbstractArray{<:Integer, 1}, 1},
#          Dofs  <: AbstractArray{<:Integer}}

#   for (n, nset) in enumerate(nsets)
#     @assert dofs[n] <= num_dofs_per_node(dof_manager)
#     update_unknown_ids!(dof_manager, nset, dofs[n])
#   end
#   resize!(dof_manager.unknown_indices, sum(dof_manager.is_unknown))
#   dof_manager.unknown_indices .= dof_ids(dof_manager)[dof_manager.is_unknown]
# end

# function update_fields!(U::NF, dof::DofManager, Uu::V) where {NF <: NodalField, V <: AbstractVector{<:Number}}
#   @assert length(Uu) == sum(dof.is_unknown)
#   for n in axes(Uu, 1)
#     U[dof.unknown_indices[n]] = Uu[n]
#   end
# end