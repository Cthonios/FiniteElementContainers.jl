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
) where {T, N, ND, NN, B}

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
  is_unknown = Trues(size(dof))
  ids = dof_ids(dof)
  dof.unknown_indices .= @views ids[is_unknown]
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

function update_unknown_ids_new!(
  dof_manager::DofManager, nsets::NSets, dofs::Dofs
) where {NSets, Dofs}

  @assert length(nsets) == length(dofs)

  # set to zero
  # dof.is_unknown .= 0
  resize!(dof_manager.unknown_indices, 0)

  # get total number of bc dofs
  n_bc_node_dofs = 0
  for n in axes(nsets, 1)
    @assert dofs[n] > 0 && dofs[n] <= num_dofs_per_node(dof_manager)
    n_bc_node_dofs += length(nsets[n])
  end

  dofs = dof_ids(dof_manager)
  # finally start pushing stuff
  # for (n, nset) in eneumate(nsets)
  #   for node in nodes
  #     push!()
  #   end
  # end
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
  # is_unknown      = BitArray(1 for _ = 1:ND, _ = 1:NN)
  is_unknown      = Trues(ND, NN)
  ids             = reshape(1:ND * NN, ND, NN)
  unknown_indices = ids[is_unknown]

  # TODO remove this once you completely rip out is_unknwon
  is_unknown = is_unknown |> collect

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
