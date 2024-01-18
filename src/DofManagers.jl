# abstract type DofManager{T, N, ND, NN, Bools} <: NodalField{T, N, ND, NN, Bools} end
# abstract type DofManager{T, N, ND, NN} end

struct DofManager{T, ND, NN, ArrType, V <: AbstractArray{T, 1}}
  unknown_dofs::V
end

function DofManager{A}(ND::Int, NN::Int) where A
  unknown_dofs = 1:ND * NN |> collect
  return DofManager{Int64, ND, NN, A, typeof(unknown_dofs)}(unknown_dofs)
end

function DofManager{ND, NN, I, ArrType}() where {ND, NN, I, ArrType}
  # unknown_dofs = Vector{Int64}(undef, 0)
  unknown_dofs = 1:ND * NN |> collect
  return DofManager{I, ND, NN, ArrType, typeof(unknown_dofs)}(unknown_dofs)
end
DofManager{ND, NN, ArrType}() where {ND, NN, ArrType} = DofManager{ND, NN, Int64, ArrType}()

Base.eltype(::DofManager{T, ND, NN, A, V}) where {T, ND, NN, A, V} = T
Base.size(::DofManager{T, ND, NN, A, V}) where {T, ND, NN, A, V} = (ND, NN)
Base.size(::DofManager{T, ND, NN, A, V}, ::Val{1}) where {T, ND, NN, A, V} = ND
Base.size(::DofManager{T, ND, NN, A, V}, ::Val{2}) where {T, ND, NN, A, V} = NN
Base.size(dof::DofManager, n::Int) = size(dof, Val(n))
num_dofs_per_node(::DofManager{T, ND, NN, A, V}) where {T, ND, NN, A, V} = ND
num_nodes(::DofManager{T, ND, NN, A, V}) where {T, ND, NN, A, V} = NN

create_fields(::DofManager{T, ND, NN, Matrix{R}, V}) where {T, ND, NN, R, V} = 
zero(SimpleNodalField{R, 2, ND, NN, Matrix{R}})
create_fields(::DofManager{T, ND, NN, Vector{R}, V}) where {T, ND, NN, R, V} = 
zero(VectorizedNodalField{R, 2, ND, NN, Vector{R}})

dof_ids(::DofManager{T, ND, NN, A, V}) where {T, ND, NN, A, V} = 1:ND * NN

create_unknowns(dof::DofManager{T, ND, NN, A, V}) where {T, ND, NN, A, V} = 
zeros(eltype(A), length(dof.unknown_dofs))

function update_fields!(U::NodalField, dof::DofManager, Uu::V) where V <: AbstractArray{<:Number, 1}
  @assert length(Uu) == length(dof.unknown_dofs)
  U[dof.unknown_dofs] = Uu
end

"""
"Default" method to reset all dofs to unknown
"""
function update_unknown_dofs!(dof::DofManager{T, ND, NN, A, V}) where {T, ND, NN, A, V}
  resize!(dof.unknown_dofs, ND * NN)
  dof.unknown_dofs .= dof_ids(dof)
end

"""
Method when all dofs are updated at once

First it resets all dofs to unknowns, then one by one sets dofs to bcs in dofs

Assumes there's a unique set of nodes provided
"""
function update_unknown_dofs!(dof_manager::DofManager, dofs::V) where V <: AbstractVector{<:Integer}
  ND = num_dofs_per_node(dof_manager)
  NN = num_nodes(dof_manager)
  resize!(dof_manager.unknown_dofs, ND * NN)
  dof_manager.unknown_dofs .= 1:ND * NN
  deleteat!(dof_manager.unknown_dofs, dofs)
end

"""
eventually use this method
"""
function bc_ids(dof_manager::DofManager, nsets::NSets, dofs::Dofs) where {NSets, Dofs}
  D = num_dofs_per_node(dof_manager)

  n_bc_nodes = 0
  for nset in nsets
    n_bc_nodes += length(nset)
  end

  bc_nodes = Vector{eltype(nsets[1])}(undef, n_bc_nodes)
  index = 1
  for n in axes(nsets, 1)
    dof = dofs[n]
    for node in nsets[n]
      bc_nodes[index] = D * (node - 1) + dof
      index += 1
    end
  end 
  sort!(unique!(bc_nodes))
end

function unknown_ids(dof_manager::DofManager, nsets::NSets, dofs::Dofs, bc_dofs) where {NSets, Dofs}
  # bc_dofs = bc_ids(dof_manager, nsets, dofs)
  dofs    = dof_ids(dof_manager) |> vec |> collect
  setdiff!(dofs, bc_dofs)
  dofs
end

# function dof_map(dof_manager::DofManager, nsets::NSets, dofs::Dofs) where {NSets, Dofs}
#   bc_dofs = bc_ids(dof_manager, nsets, dofs)
#   unknown_dofs = unknown_ids(dof_manager, nsets, dofs, bc_dofs)

#   all_dofs = Vector{eltype(dofs)}(undef, length(unknown_dofs) + length(bc_dofs))
# end

# function unknown_ids(dof_manager::DofManager, nsets::NSets, dofs::Dofs) where {NSets, Dofs}
#   ids = dof_ids(dof_manager)
#   # temp = Hcat(nsets)
#   # unique(temp)

#   temp = Vcat(nsets...)
#   # temp = nsets[1]
#   # for n in 2:length(nsets)
#   #   # unique!(Hcat(temp, nsets[n]))
#   #   temp = Vcat(temp, nsets[n])
#   # end
# end

##########################################################################

# struct SimpleDofManager{
#   T, N, ND, NN, Bools <: BitArray{2}, Indices <: AbstractArray{<:Integer, 1}
# } <: DofManager{T, N, ND, NN, Bools}
#   is_unknown::Bools
#   unknown_dofs::Indices
# end


######################################################

# struct SimpleDofManager{
#   T, N, ND, NN, Bools <: AbstractArray{Bool, 2}, Indices <: AbstractArray{<:Integer, 1}
# } <: DofManager{T, N, ND, NN, Bools}
#   is_unknown::Bools
#   unknown_dofs::Indices
# end

# Base.IndexStyle(::Type{<:SimpleDofManager}) = IndexLinear()

# Base.axes(field::SimpleDofManager) = axes(field.is_unknown)
# Base.getindex(field::SimpleDofManager, n::Int) = getindex(field.is_unknown, n)
# Base.setindex!(field::SimpleDofManager, v, n::Int) = setindex!(field.is_unknown, v, n)
# Base.size(::SimpleDofManager{T, N, ND, NN, V}) where {T, N, ND, NN, V} = (ND, NN)


# function SimpleDofManager{ND, NN}() where {ND, NN}
#   # is_unknown      = BitArray(1 for _ = 1:ND, _ = 1:NN)
#   is_unknown      = Trues(ND, NN)
#   ids             = reshape(1:ND * NN, ND, NN)
#   unknown_dofs = ids[is_unknown]

#   # TODO remove this once you completely rip out is_unknwon
#   is_unknown = is_unknown |> collect

#   return SimpleDofManager{Bool, 2, ND, NN, typeof(is_unknown), typeof(unknown_dofs)}(is_unknown, unknown_dofs)
# end

# ##########################################################################

# struct VectorizedDofManager{
#   T, N, ND, NN, Bools <: AbstractArray{Bool, 1}, Indices <: AbstractArray{<:Integer, 1}
# } <: DofManager{T, N, ND, NN, Bools}
#   is_unknown::Bools
#   unknown_dofs::Indices
# end

# Base.IndexStyle(::Type{<:VectorizedDofManager}) = IndexLinear()
# Base.axes(dof::VectorizedDofManager) = (Base.OneTo(num_dofs_per_node(dof)), Base.OneTo(num_nodes(dof)))
# Base.getindex(dof::VectorizedDofManager, n::Int) = getindex(dof.is_unknown, n)
# function Base.getindex(dof::VectorizedDofManager, d::Int, n::Int) 
#   @assert d > 0 && d <= num_dofs_per_node(dof)
#   @assert n > 0 && n <= num_nodes(dof)
#   getindex(dof.is_unknown, (n - 1) * num_dofs_per_node(dof) + d)
# end
# Base.setindex!(dof::VectorizedDofManager, v, n::Int) = setindex!(dof.is_unknown, v, n)
# function Base.setindex!(dof::VectorizedDofManager, v, d::Int, n::Int)
#   @assert d > 0 && d <= num_dofs_per_node(dof)
#   @assert n > 0 && n <= num_nodes(dof)
#   setindex!(dof.is_unknown, v, (n - 1) * num_dofs_per_node(dof) + d)
# end
# Base.size(::VectorizedDofManager{T, N, ND, NN, V}) where {T, N, ND, NN, V} = (ND, NN)

# function VectorizedDofManager{ND, NN}() where {ND, NN}
#   is_unknown      = BitVector(1 for _ = 1:ND * NN)
#   ids             = 1:ND * NN
#   unknown_dofs = ids[is_unknown]
#   return VectorizedDofManager{Bool, 2, ND, NN, typeof(is_unknown), typeof(unknown_dofs)}(is_unknown, unknown_dofs)
# end

# ##########################################################################

# DofManager{ND, NN, Matrix}() where {ND, NN} = SimpleDofManager{ND, NN}()
# DofManager{ND, NN, Vector}() where {ND, NN} = VectorizedDofManager{ND, NN}()
