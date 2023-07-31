function number_of_hessian_entries(mesh::Mesh, is_unknown::BitMatrix)
  n_hessian_entries::Int = 0
  for block in mesh.blocks
    unknowns = is_unknown[:, block.conn]
    for e in axes(unknowns, 3)
      n_hessian_entries = @views n_hessian_entries + sum(unknowns[:, :, e])^2
    end 
  end
  return n_hessian_entries
end

function number_of_hessian_entries_naive(mesh::Mesh)
  n_hessian_entries::Int = 0
  for block in mesh.blocks
    n_nodes_per_elem, n_elem = size(block.conn)
    n_dim = size(mesh.coords, 1)
    n_hessian_entries = n_hessian_entries + n_elem * (n_dim * n_nodes_per_elem)^2
  end
  return n_hessian_entries
end

# function setup_hessian_bc_masks!(bc_masks::Vector{Array{Bool, 3}}, mesh::Mesh, is_bc::BitMatrix)
# function setup_hessian_bc_masks!(bc_masks::Vector{BitArray{3}}, mesh::Mesh, is_bc::BitMatrix)
#   for (n, block) in enumerate(mesh.blocks)
#     is_bc_block = @views is_bc[:, block.conn]
#     bc_masks[n] .= true 
    
#   end
# end

# function setup_hessian_bc_masks(mesh::Mesh, is_bc::BitMatrix)
#   # bc_masks = Vector{Array{Bool, 3}}(undef, length(mesh.blocks))
#   bc_masks = Vector{BitArray{3}}(undef, length(mesh.blocks))
#   n_fields = size(is_bc, 1)
#   for (n, block) in enumerate(mesh.blocks)
#     n_nodes_per_elem, n_elem = size(block.conn)
#     n_dof_per_elem = n_fields * n_nodes_per_elem
#     # bc_masks[n] = Array{Bool, 3}(undef, n_dof_per_elem, n_dof_per_elem, n_elem)
#     bc_masks[n] = BitArray{3}(undef, n_dof_per_elem, n_dof_per_elem, n_elem)
#   end
#   setup_hessian_bc_masks!(bc_masks, mesh, is_bc)
#   # display(bc_masks)
# end

function setup_hessian_coordinates!(row_coords::Vector{Itype}, col_coords::Vector{Itype}, mesh::Mesh, ids::Matrix{<:Integer}) where Itype
  counter = 1

  for block in mesh.blocks
    conn = block.conn
    conn = convert(Matrix{Itype}, conn)

    for e in axes(conn, 2)
      ids_e = @views vec(ids[:, conn[:, e]])
      for i in axes(ids_e, 1)
        for j in axes(ids_e, 1)
          row_coords[counter] = ids_e[i]
          col_coords[counter] = ids_e[j]
          counter = counter + 1
        end
      end

    end
  end
end

struct DofManager{Itype, D, Rtype}
  field_shape::Tuple{Int, Int}
  is_bc::BitMatrix
  bc_indices::Vector{Itype}
  is_unknown::BitMatrix
  unknown_indices::Vector{Itype}
  essential_bcs::Vector{EssentialBC{Itype, D, Rtype}}
  row_coords::Vector{Itype}
  col_coords::Vector{Itype}
end
# Base.length(d::DofManager) = d.field_shape[1] * d.field_shape[2]

create_fields(d::DofManager{Itype, D, Rtype}) where {Itype, D, Rtype} = zeros(Rtype, d.field_shape)
create_unknowns(d::DofManager{Itype, D, Rtype}) where {Itype, D, Rtype} = zeros(Rtype, length(d.unknown_indices))

function create_linear_system(d::DofManager{Itype, D, Rtype}) where {Itype, D, Rtype}
  R = Vector{Rtype}(undef, length(d.row_coords))
  K = sparse(d.row_coords, d.col_coords, zeros(Rtype, length(d.row_coords)))
  return R, K
end

f_zero(::SVector{D, Rtype}) where {D, Rtype} = 0.0
f_zero(::SVector{D, Rtype}, t::Rtype) where {D, Rtype} = 0.0

"""
for zero dirichlet bc
"""
function update_bc!(U::Matrix{Rtype}, d::DofManager{Itype, D, Rtype}, bc_index::Int) where {Itype, D, Rtype}
  bc = d.essential_bcs[bc_index]
  @inbounds U[bc.dof, bc.nodes] .= f_zero.(bc.coords)
end

"""
for time in-dependent bcs
"""
function update_bc!(U::Matrix{Rtype}, d::DofManager{Itype, D, Rtype}, bc_index::Int, f::Function) where {Itype, D, Rtype}
  bc = d.essential_bcs[bc_index]
  @views map!(x -> f(x, 0.), U[bc.dof, bc.nodes], bc.coords)
end

"""
for time dependent bcs
"""
function update_bc!(U::Matrix{Rtype}, d::DofManager{Itype, D, Rtype}, bc_index::Int, f::Function, t::Rtype) where {Itype, D, Rtype}
  bc = d.essential_bcs[bc_index]
  @views map!(x -> f(x, t), U[bc.dof, bc.nodes], bc.coords)
end

function update_fields!(U::Matrix{Rtype}, d::DofManager{Itype, D, Rtype}, Uu::Vector{Rtype}) where {Itype, D, Rtype}
  @assert length(Uu) == length(d.unknown_indices)
  @inbounds U[d.is_unknown] = Uu
end

function DofManager(
  mesh::Mesh,
  n_dofs::Int,
  essential_bcs::Vector{EssentialBC{Itype, D, Rtype}},
) where {Itype, D, Rtype}

  field_shape = n_dofs, size(mesh.coords, 2)

  is_bc = BitMatrix(undef, field_shape)
  is_bc .= 0

  for bc in essential_bcs
    is_bc[bc.dof, bc.nodes] .= 1
  end

  is_unknown = .!is_bc
  ids = reshape(1:length(is_bc), field_shape) |> collect

  unknown_indices = ids[is_unknown]
  bc_indices = ids[is_bc]

  n_hessian_entries = number_of_hessian_entries_naive(mesh)

  # @time bc_masks = setup_hessian_bc_masks(mesh, is_bc)

  col_coords = Vector{Itype}(undef, n_hessian_entries)
  row_coords = Vector{Itype}(undef, n_hessian_entries)

  setup_hessian_coordinates!(col_coords, row_coords, mesh, ids)

  # display(row_coords)
  # display(col_coords)

  return DofManager{Itype, D, Rtype}(field_shape, 
                                     is_bc, bc_indices, 
                                     is_unknown, unknown_indices, 
                                     essential_bcs,
                                     row_coords, col_coords)
end