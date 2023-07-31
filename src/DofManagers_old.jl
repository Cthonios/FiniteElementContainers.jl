function setup_hessian_coordinates(
  conns::Vector{Matrix{Itype}},
  ids::Matrix{<:Integer}, is_unknown::Matrix{Bool}, dof_to_unknown::Vector{<:Integer}
) where Itype

  n_els = sum(map(x -> size(x, 2), conns))

  n_el_unknowns = zeros(Int, n_els)
  n_hessian_entries = 0
  el_count = 1
  for conn in conns
    for e in axes(conn, 2)
      el_unknown_flags = @views is_unknown[:, conn[:, e]]
      n_el_unknowns[el_count] = sum(el_unknown_flags)
      n_hessian_entries = n_hessian_entries + n_el_unknowns[e]^2
      el_count = el_count + 1
    end
  end
  
  row_coords = zeros(Int, n_hessian_entries)
  col_coords = zeros(Int, n_hessian_entries)

  range_begin = 1
  el_count = 1

  @views begin
    for conn in conns
      for e in axes(conn, 2)
        el_dofs = ids[:, conn[:, e]]
        el_unknown_flags = is_unknown[:, conn[:, e]]
        el_unknowns = dof_to_unknown[el_dofs[el_unknown_flags]]
        el_hess_coords = repeat(el_unknowns, 1, n_el_unknowns[el_count])

        range_end = range_begin + n_el_unknowns[el_count]^2

        row_coords[range_begin:range_end - 1] = el_hess_coords[:]
        col_coords[range_begin:range_end - 1] = el_hess_coords'[:]

        range_begin = range_begin + n_el_unknowns[el_count]^2
        el_count = el_count + 1
      end
    end
  end

  return row_coords, col_coords
end

function setup_hessian_coordinates_v2(
  conns::Vector{Matrix{Itype}},
  ids::Matrix{<:Integer}, is_unknown::Matrix{Bool}, dof_to_unknown::Vector{<:Integer}
) where Itype

  n_els = sum(map(x -> size(x, 2), conns))

  n_el_unknowns = zeros(Int, n_els)
  n_hessian_entries = 0
  el_count = 1
  for conn in conns
    for e in axes(conn, 2)
      el_unknown_flags = @views is_unknown[:, conn[:, e]]
      n_el_unknowns[el_count] = sum(el_unknown_flags)
      n_hessian_entries = n_hessian_entries + n_el_unknowns[e]^2
      el_count = el_count + 1
    end
  end
  
  row_coords = zeros(Int, n_hessian_entries)
  col_coords = zeros(Int, n_hessian_entries)

  range_begin = 1
  el_count = 1

  # trying to allocate these upfront
  for conn in conns
    el_dofs = @views ids[:, conn]
    el_unknown_flags = @views is_unknown[:, conn]
    el_dofs_to_unknown = @views el_dofs[el_unknown_flags]
    # el_unknowns = @views dof_to_unknown[el_dofs_to_unknown]
    @show size(el_dofs_to_unknown)

    for e in axes(conn, 2)
      # # el_dofs_to_unknown[:, :, e] |> display
      # el_dofs |> display
      # el_unknown_flags |> display
      # @time temp =
      # el_unknowns = dof_to_unknown[]
    end
    # # @show el_unknowns
    # # @show size(el_unknowns)
    # @show size(el_dofs)
    # @show size(el_unknown_flags)
    # @show size(el_dofs_to_unknown)
    # @show size(el_unknowns)
    # # @show ids[:, conn[:, 1]]
    # ids[:, conn[:, 10]] |> display
    # el_dofs[:, :, 10] |> display

    # is_unknown[:, conn[:, 2000]] |> display
    # el_unknown_flags[:, :, 2000] |> display

    # dof_to_unknown[ids[:, conn[:, 10]][is_unknown[:, conn[:, 10]]]] |> display
    # # el_unknowns



    # for e in axes(conn, 2)
    #   el_hess_coords = repeat(el_unknowns, 1, n_el_unknowns[el_count])

    #   range_end = range_begin + n_el_unknowns[el_count]^2

    #   row_coords[range_begin:range_end - 1] = el_hess_coords[:]
    #   col_coords[range_begin:range_end - 1] = el_hess_coords'[:]

    #   range_begin = range_begin + n_el_unknowns[el_count]^2
    #   el_count = el_count + 1
    # end
  end

  # for conn in conns
  #   for e in axes(conn, 2)
  #     el_dofs = ids[:, conn[:, e]]
  #     el_unknown_flags = is_unknown[:, conn[:, e]]
  #     el_unknowns = dof_to_unknown[el_dofs[el_unknown_flags]]
  #     el_hess_coords = repeat(el_unknowns, 1, n_el_unknowns[el_count])

  #     range_end = range_begin + n_el_unknowns[el_count]^2

  #     row_coords[range_begin:range_end - 1] = el_hess_coords[:]
  #     col_coords[range_begin:range_end - 1] = el_hess_coords'[:]

  #     range_begin = range_begin + n_el_unknowns[el_count]^2
  #     el_count = el_count + 1
  #   end
  # end

  return row_coords, col_coords
end

function setup_hessian_bc_mask(
  # fspace::FunctionSpace{Itype, N, D, Ftype, L, Q},
  # fspace::T,
  conn::Matrix{<:Integer},
  ids::Matrix{<:Integer}, is_bc::Matrix{<:Integer}
# ) where {Itype, N, D, Ftype, L, Q}
)

  # n_els = length(fspace)  
  # n_els = size(fspace, 2)
  n_els = size(conn, 2)
  # n_nodes_per_element = size(fspace.connectivity, 1)
  n_nodes_per_element = size(conn, 1)
  n_fields = size(ids, 1)
  n_dofs_per_e = n_nodes_per_element * n_fields
  hessian_bc_mask = Array{Bool, 3}(undef, n_dofs_per_e, n_dofs_per_e, n_els)
  hessian_bc_mask .= true

  # for e in 1:length(fspace)
  # for e in 1:size(fspace, 2)
  for e in axes(conn, 2)
    # e_flag = @views is_bc[:, connectivity(fspace, e)][:]
    e_flag = @views is_bc[:, conn[:, e]][:]
    hessian_bc_mask[e_flag, :, e] .= false
    hessian_bc_mask[:, e_flag, e] .= false
  end

  return hessian_bc_mask
end

# TODO type on Int
struct DofManager
  field_shape::Tuple{Int, Int}
  is_bc::Matrix{Bool}
  is_unknown::Matrix{Bool}
  ids::Matrix{Int}
  bc_indices::Vector{Int}
  unknown_indices::Vector{Int}
  dof_to_unknown::Vector{Int}
  row_coords::Vector{Int}
  col_coords::Vector{Int}
  bc_masks::Vector{Array{Bool, 3}}
  block_connectivities::Vector{Matrix{Int}}
end

create_field(d::DofManager, ::Type{T} = Float64) where T = zeros(T, d.field_shape)
get_bc_size(d::DofManager) = sum(d.is_bc)
get_bc_values(d::DofManager, U::A) where A <: AbstractArray = @views U[d.bc_indices]
get_unknown_size(d::DofManager) = sum(d.is_unknown)
get_unknown_values(d::DofManager, U::A) where A <: AbstractArray = @views U[d.unknown_indices]

function DofManager(
  mesh::Mesh{B, F},
  essential_bcs::Vector{EssentialBC},
  n_dofs::Integer
) where {B, F}
  field_shape = n_dofs, size(mesh.coords, 2)

  is_bc = Matrix{Bool}(undef, field_shape)
  is_bc .= false

  for bc in essential_bcs
    nset = filter(x -> x.id == bc.nset_id, mesh.nsets)
    if length(nset) < 0
      throw(BoundsError("nset id not found"))
    else
      nset = nset[1]
    end
    is_bc[bc.dof, nset.nodes] .= true
  end
  is_unknown = map(x -> !x, is_bc)
  ids = reshape(1:length(is_bc), field_shape) |> collect

  unknown_indices = ids[is_unknown]
  bc_indices = ids[is_bc]

  dof_to_unknown = -1 * ones(Int, length(is_bc))
  dof_to_unknown[unknown_indices] = 1:length(unknown_indices)

  block_connectivities = Vector{Matrix{Int}}(undef, length(mesh.blocks))
  for (n, block) in enumerate(mesh.blocks)
    block_connectivities[n] = block.conn
  end

  # row_coords, col_coords = setup_hessian_coordinates(fspaces, ids, is_unknown, dof_to_unknown)
  # @time row_coords, col_coords = setup_hessian_coordinates(block_connectivities, ids, is_unknown, dof_to_unknown)
  @time row_coords, col_coords = setup_hessian_coordinates_v2(block_connectivities, ids, is_unknown, dof_to_unknown)


  # bc_masks = Vector{Array{Bool, 3}}(undef, length(fspaces))
  # for (n, fspace) in enumerate(fspaces)
  #   bc_masks[n] = setup_hessian_bc_mask(fspace, ids, is_bc)
  # end

  bc_masks = Vector{Array{Bool, 3}}(undef, length(block_connectivities))
  for (n, conn) in enumerate(block_connectivities)
    bc_masks[n] = setup_hessian_bc_mask(conn, ids, is_bc)
  end

  return DofManager(
    field_shape, 
    is_bc, is_unknown, 
    ids, bc_indices, unknown_indices,
    dof_to_unknown,
    row_coords, col_coords, bc_masks,
    block_connectivities
  )
end
