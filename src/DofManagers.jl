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

function number_of_hessian_entries_naive(mesh::Mesh, ids::VecOrMat{<:Integer})
  n_hessian_entries::Int = 0
  if typeof(ids) <: Matrix
    n_dof = size(ids, 1)
  else
    n_dof = 1
  end

  # for block in mesh.blocks
  for block in values(mesh.blocks)
    n_nodes_per_elem, n_elem = size(block.conn)
    n_hessian_entries = n_hessian_entries + n_elem * (n_dof * n_nodes_per_elem)^2
  end

  return n_hessian_entries
end

function setup_hessian_coordinates!(row_coords::Vector{Itype}, col_coords::Vector{Itype}, mesh::Mesh, ids::VecOrMat{<:Integer}) where Itype
  counter = 1

  # for block in mesh.blocks
  for block in values(mesh.blocks)
    conn = block.conn
    conn = convert(Matrix{Itype}, conn)

    for e in axes(conn, 2)
      if typeof(ids) <: Matrix
        ids_e = @views vec(ids[:, conn[:, e]])
      else
        ids_e = @view conn[:, e]
      end
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

struct DofManager{Itype, Rtype, NDof}
  field_shape::Union{Int, Tuple{Int, Int}}
  # field_shape::Union{Itype, Tuple{Itype, Itype}}
  # ids::Matrix{Itype}
  ids::VecOrMat{Itype}
  is_bc::BitArray{NDof}
  bc_indices::Vector{Itype}
  is_unknown::BitArray{NDof}
  unknown_indices::Vector{Itype}
  essential_bcs::Vector{EssentialBC{Itype, Rtype}}
  row_coords::Vector{Itype}
  col_coords::Vector{Itype}
end

create_fields(d::DofManager{Itype, Rtype, NDof}) where {Itype, Rtype, NDof} = zeros(Rtype, d.field_shape)
create_unknowns(d::DofManager{Itype, Rtype, NDof}) where {Itype, Rtype, NDof} = zeros(Rtype, length(d.unknown_indices))
ndofs(::DofManager{Itype, Rtype, NDof}) where {Itype, Rtype, NDof} = NDof

f_zero() = 0.0
f_zero(::SVector{D, Rtype}) where {D, Rtype} = 0.0
f_zero(::SVector{D, Rtype}, t::Rtype) where {D, Rtype} = 0.0

"""
for zero dirichlet bc
"""
function update_bc!(U::Vector{Rtype}, d::DofManager{Itype, Rtype, NDof}, bc_index::Int) where {Itype, Rtype, NDof}
  bc = d.essential_bcs[bc_index]
  @inbounds U[bc.nodes] .= f_zero.(bc.coords)
end

function update_bcs!(U::Vector{Rtype}, d::DofManager{Itype, Rtype, NDof}) where {Itype, Rtype, NDof}
  for bc in d.essential_bcs
    # @inbounds U[bc.nodes] .= f_zero.(bc.coords)
    @inbounds U[bc.nodes] .= f_zero()
  end
end

function update_bc!(U::Matrix{Rtype}, d::DofManager{Itype, Rtype, NDof}, bc_index::Int) where {Itype, Rtype, NDof}
  bc = d.essential_bcs[bc_index]
  @inbounds U[bc.dof, bc.nodes] .= f_zero.(bc.coords)
end

"""
for time in-dependent bcs
"""
function update_bc!(U::Vector{Rtype}, d::DofManager{Itype, Rtype, NDof}, bc_index::Int, f::Function) where {Itype, Rtype, NDof}
  bc = d.essential_bcs[bc_index]
  @views map!(x -> f(x, 0.), U[bc.nodes], bc.coords)
end

function update_bc!(U::Matrix{Rtype}, d::DofManager{Itype, Rtype, NDof}, bc_index::Int, f::Function) where {Itype, Rtype, NDof}
  bc = d.essential_bcs[bc_index]
  @views map!(x -> f(x, 0.), U[bc.dof, bc.nodes], bc.coords)
end

"""
for time dependent bcs
"""
function update_bc!(U::Vector{Rtype}, d::DofManager{Itype, Rtype, NDof}, bc_index::Int, f::Function, t::Rtype) where {Itype, Rtype, NDof}
  bc = d.essential_bcs[bc_index]
  @views map!(x -> f(x, t), U[bc.nodes], bc.coords)
end

function update_bc!(U::Matrix{Rtype}, d::DofManager{Itype, Rtype, NDof}, bc_index::Int, f::Function, t::Rtype) where {Itype, Rtype, NDof}
  bc = d.essential_bcs[bc_index]
  @views map!(x -> f(x, t), U[bc.dof, bc.nodes], bc.coords)
end

"""
"""
function update_fields!(U::VecOrMat{Rtype}, d::DofManager{Itype, Rtype, NDof}, Uu::Vector{Rtype}) where {Itype, Rtype, NDof}
  @assert length(Uu) == length(d.unknown_indices)
  @inbounds U[d.is_unknown] = Uu
end

function DofManager(
  mesh::Mesh{F, I, B},
  n_dofs::Int,
  # essential_bcs::Vector{EssentialBC{B, D, F}},
  essential_bcs::Vector{<:EssentialBC}
) where {F, I, B}

  if n_dofs == 1
    field_shape = size(mesh.coords, 2)
  else
    field_shape = n_dofs, size(mesh.coords, 2)
  end

  D = size(mesh.coords, 1)

  is_bc = BitArray(undef, field_shape)
  is_bc .= 0

  for bc in essential_bcs
    if n_dofs == 1
      is_bc[bc.nodes] .= 1
    else
      is_bc[bc.dof, bc.nodes] .= 1
    end
  end

  is_unknown = .!is_bc
  ids = reshape(1:length(is_bc), field_shape) |> collect
  ids = convert.(B, ids)
  
  unknown_indices = ids[is_unknown]
  bc_indices = ids[is_bc]

  n_hessian_entries = number_of_hessian_entries_naive(mesh, ids)

  col_coords = Vector{B}(undef, n_hessian_entries)
  row_coords = Vector{B}(undef, n_hessian_entries)

  setup_hessian_coordinates!(col_coords, row_coords, mesh, ids)

  return DofManager{B, Float64, n_dofs}(field_shape, ids,
                                        is_bc, bc_indices, 
                                        is_unknown, unknown_indices, 
                                        essential_bcs,
                                        row_coords, col_coords)
end