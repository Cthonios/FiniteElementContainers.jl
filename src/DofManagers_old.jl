# """
# """
# struct Connectivity{T <: AbstractArray{<:Integer}} #<: AbstractArray{T, 1}
#   conn::T
# end

# Base.length(c::Connectivity) = length(c.conn)
# Base.axes(c::Connectivity) = Base.OneTo(length(c))
# Base.getindex(c::Connectivity, i::Int) = c.conn[i]

struct Connectivity{V <: AbstractVector{<:Integer}}
  nodes::V
  num_nodes_per_elem::V
end

function Connectivity(blocks::Vector{MeshBlock{T, Matrix{T}}}) where T
  nodes = T[]
  num_nodes_per_elem = T[]
  for block in blocks
    append!(nodes, block.conn[:])
    append!(num_nodes_per_elem, fill(size(block.conn, 1), size(block.conn, 2)))
  end
  return Connectivity(nodes, num_nodes_per_elem)
end

Base.length(c::C) where C <: Connectivity = length(c.num_nodes_per_elem)
Base.getindex(c::C, e::Int) where C <: Connectivity = c.nodes[]

"""
"""
struct DofManager{
  NDof, 
  B <: AbstractArray{Bool, 2},
  V <: AbstractArray{<:Integer},
  S <: StructArray{<:Connectivity{<:Vector{<:Integer}}}
}
	is_unknown::B
	unknown_indices::V
  conns::S
  dof_conns::S
end

"""
"""
n_dofs(::DofManager{NDof, B, V, S}) where {NDof, B, V, S} = NDof

"""
"""
function DofManager(mesh::M, n_dofs::Int) where M <: Mesh
  # convenient parameters
  n_nodes = size(mesh.coords, 2)
  n_els   = map(x -> size(x.conn, 2), mesh.blocks) |> sum
  
  # setup indexing arrays
  is_unknown      = BitArray(1 for _ = 1:n_dofs, _ = 1:n_nodes)
  ids             = reshape(1:length(is_unknown), n_dofs, n_nodes)
  unknown_indices = ids[is_unknown]


end

"""
"""
function DofManager_old(
  mesh::Mesh{M, V1, V2, V3},
  n_dofs::Int
) where {M, V1, V2, V3}

  # convenient parameters
  n_nodes = size(mesh.coords, 2)
  n_els   = map(x -> size(x.conn, 2), mesh.blocks) |> sum

  # setup indexing arrays
  is_unknown      = BitArray(1 for _ = 1:n_dofs, _ = 1:n_nodes)
  ids             = reshape(1:length(is_unknown), n_dofs, n_nodes)
  unknown_indices = ids[is_unknown]

  # using a struct array - TODO template this
  conns     = StructArray{Connectivity{Vector{Int64}}}(undef, n_els)
  dof_conns = StructArray{Connectivity{Vector{Int64}}}(undef, n_els)
  n = 1
  for block in mesh.blocks
    for e in axes(block.conn, 2)
      # TODO for typing below
      @views conns[n] = Connectivity(convert.(Int64, block.conn[:, e]))
      @views dof_conns[n] = Connectivity(convert.(Int64, vec(ids[:, block.conn[:, e]])))
      n = n + 1
    end
  end

  # get types
  B = typeof(is_unknown)
  V = typeof(unknown_indices)
  S = typeof(conns)
  return DofManager{n_dofs, B, V, S}(is_unknown, unknown_indices, conns, dof_conns)
end

"""
"""
create_fields(d::DofManager, Rtype::Type{<:AbstractFloat} = Float64) = zeros(Rtype, size(d.is_unknown))

"""
"""
create_unknowns(d::DofManager, Rtype::Type{<:AbstractFloat} = Float64) = zeros(Rtype, sum(d.is_unknown))

"""
"""
dof_ids(d::DofManager) = reshape(1:length(d.is_unknown), size(d.is_unknown))

"""
"""
Base.size(d::DofManager) = size(d.is_unknown)

"""
"""
element_connectivity(d::DofManager, e::Int) = d.conns[e].conn

"""
"""
element_connectivity_2(d::DofManager{NDOF, B, V, S}, e::Int) where {NDOF, B, V, S} = d.dof_conns[e].conn[1:NDOF:end]

"""
"""
dof_connectivity(d::DofManager, e::Int) = d.dof_conns[e].conn


function update_bcs!(
  U::M1,
  mesh::Mesh{M2, V1, V2, V3},
  dof::DofManager,
  bcs::V4
) where {M1 <: AbstractMatrix, M2 <: AbstractMatrix,
         V1 <: AbstractVector, V2 <: AbstractVector,
         V3 <: AbstractVector, V4 <: AbstractVector{<:EssentialBC}}

  dof.is_unknown .= 1
  for bc in bcs
    for node in bc.nodes
      dof.is_unknown[bc.dof, node] = 0
      U[bc.dof, node] = @views bc.func(mesh.coords[:, node], 0.)
    end
  end

  # TODO below line is the only source of allocations here
  @views new_unknown_indices = dof_ids(dof)[dof.is_unknown]
	resize!(dof.unknown_indices, length(new_unknown_indices))
	dof.unknown_indices .= new_unknown_indices
end

function update_fields!(U::M, d::DofManager, Uu::V) where {M <: AbstractMatrix, V <: AbstractVector}
  @assert length(Uu) == sum(d.is_unknown)
  U[d.is_unknown] = Uu
end 