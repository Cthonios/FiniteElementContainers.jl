struct DofManager{Itype, Rtype, NDof}
  field_shape::Union{Int, Tuple{Int, Int}}
  ids::VecOrMat{Itype}
  is_unknown::BitArray
  unknown_indices::Vector{Itype}
  conns::Vector{Matrix{Itype}}
end

create_fields(d::DofManager{Itype, Rtype, NDof}) where {Itype, Rtype, NDof} = zeros(Rtype, d.field_shape)
create_unknowns(d::DofManager{Itype, Rtype, NDof}) where {Itype, Rtype, NDof} = zeros(Rtype, length(d.unknown_indices))
ndofs(::DofManager{Itype, Rtype, NDof}) where {Itype, Rtype, NDof} = NDof

f_zero() = 0.0
f_zero(::SVector{D, Rtype}) where {D, Rtype} = 0.0
f_zero(::SVector{D, Rtype}, t::Rtype) where {D, Rtype} = 0.0

function DofManager(
  mesh::Mesh{F, I, B},
  n_dofs::Int,
  essential_bcs::Vector{<:EssentialBC}
) where {F, I, B}

  # Note this implementation assumes you are using
  # all of the blocks in the exodus file

  if n_dofs == 1
    field_shape = size(mesh.coords, 2)
  else
    field_shape = n_dofs, size(mesh.coords, 2)
  end

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

  conns = Vector{Matrix{B}}(undef, length(mesh.blocks))
  for (n, block) in enumerate(mesh.blocks)
    conn = block.conn
    if n_dofs > 1
      temp_conn = Matrix{B}(undef, n_dofs * size(conn, 1), size(conn, 2))
      for e in axes(conn, 2)
        ids_e = @views vec(ids[:, conn[:, e]])
        temp_conn[:, e] = ids_e
      end
      conns[n] = temp_conn
    else
      conns[n] = conn
    end
  end

  return DofManager{B, F, n_dofs}(field_shape, ids, is_unknown, unknown_indices, conns)
end

function number_of_hessian_entries_naive(dof::DofManager{Itype, Rtype, NDof}) where {Itype, Rtype, NDof}

  n_hessian_entries::Int = 0
  if typeof(dof.ids) <: Matrix
    n_dof = size(dof.ids, 1)
  else
    n_dof = 1
  end

  # for block in mesh.blocks
  # for block in values(mesh.blocks)
  for conn in dof.conns
    # n_nodes_per_elem, n_elem = size(block.conn)
    n_nodes_per_elem, n_elem = size(conn)
    # n_hessian_entries = n_hessian_entries + n_elem * (n_dof * n_nodes_per_elem)^2
    n_hessian_entries = n_hessian_entries + n_elem * n_nodes_per_elem^2

  end

  return n_hessian_entries
end

function setup_hessian_coordinates!(
  row_coords::Vector{Itype}, 
  col_coords::Vector{Itype}, 
  dof::DofManager{Itype, Rtype, NDof}
) where {Itype, Rtype, NDof}

  counter = 1

  for conn in values(dof.conns)
    conn = convert(Matrix{Itype}, conn)

    for e in axes(conn, 2)
      ids_e = @views conn[:, e]

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

# TODO need to update update_bcs to update is_unknown array if bcs change

function update_bcs!(
  U::Vector{Rtype}, 
  mesh::Mesh,
  bcs::Vector{<:EssentialBC}
) where Rtype

  for bc in bcs
    # @inbounds U[bc.nodes] .= f_zero()
    @inbounds U[bc.nodes] .= @views bc.func.(eachcol(mesh.coords[:, bc.nodes]), (0.,))
  end
end

"""
for static problems with more than one dof
"""
function update_bcs!(
  U::Matrix{Rtype},
  mesh::Mesh{Rtype, I, B},
  bcs::Vector{<:EssentialBC}
) where {Rtype, I, B}

  for bc in bcs
    @inbounds U[bc.dof, bc.nodes] .= @views bc.func.(eachcol(mesh.coords[:, bc.nodes]), (0.,))
  end
end

function update_fields!(U::VecOrMat{Rtype}, d::DofManager{Itype, Rtype, NDof}, Uu::Vector{Rtype}) where {Itype, Rtype, NDof}
  @assert length(Uu) == length(d.unknown_indices)
  U[d.is_unknown] = Uu
end
