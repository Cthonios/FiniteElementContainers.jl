function setup_mesh_arrays!(
  blocks::Vector{Block{B}}, nsets::Vector{NodeSet{B}}, ssets::Vector{SideSet{B}},
  exo::ExodusDatabase{M, I, B, F},
  block_ids::Vector{<:Integer}, nset_ids::Vector{<:Integer}, sset_ids::Vector{<:Integer}
) where {M, I, B, F}
  for (n, block_id) in enumerate(block_ids)
    blocks[n] = Block(exo, block_id)
  end

  for (n, nset_id) in enumerate(nset_ids)
    nsets[n] = NodeSet(exo, nset_id)
  end

  for (n, sset_id) in enumerate(sset_ids)
    ssets[n] = SideSet(exo, sset_id)
  end
end

struct Mesh{B, F}
  coords::Matrix{F}
  blocks::Vector{Block{B}}
  nsets::Vector{NodeSet{B}}
  ssets::Vector{SideSet{B}}
end

function Mesh(
  exo::ExodusDatabase{M, I, B, F},
  block_ids::Vector{<:Integer},
  nset_ids::Vector{<:Integer} = Int[],
  sset_ids::Vector{<:Integer} = Int[]
) where {M, I, B, F}

  coords = read_coordinates(exo)

  blocks = Vector{Block{B}}(undef, length(block_ids))
  nsets  = Vector{NodeSet{B}}(undef, length(nset_ids))
  ssets  = Vector{SideSet{B}}(undef, length(sset_ids))

  setup_mesh_arrays!(blocks, nsets, ssets, exo, block_ids, nset_ids, sset_ids)

  return Mesh{B, F}(coords, blocks, nsets, ssets)
end

function Mesh(n_x, n_y, x_extent, y_extent)
  e_x, e_y = n_x - 1, n_y - 1

  xs = LinRange(x_extent[1], x_extent[2], n_x)
  ys = LinRange(y_extent[1], y_extent[2], n_y)
  coords = Matrix{Float64}(undef, 2, n_x * n_y)

  n = 1
  for i in 1:n_x
    for j in 1:n_y
      coords[1, n] = xs[i]
      coords[2, n] = ys[j]
      n = n + 1
    end
  end

  conns = Matrix{Int64}(undef, 3, 2 * e_x * e_y)

  e = 1
  for i in 0:e_x - 1
    for j in 0:e_y - 1
      conns[1, e] = i + n_x * j
      conns[2, e] = i + 1 + n_x * j
      conns[3, e] = i + 1 + n_x * (j + 1)
      #
      e = e + 1
      conns[1, e] = i + n_x * j
      conns[2, e] = i + 1 + n_x * (j + 1)
      conns[3, e] = i + n_x * (j + 1)
      #
      e = e + 1
    end
  end
  conns = conns .+ 1
  block = Block(1, size(conns, 2), 3, "tri3", conns)
  blocks = [block]

  # top_coords = filter(x -> x[2] - 1.0 < eps(), coords)
  top_coords = findall(x -> x[2] ≈ 1.0, eachcol(coords))
  bottom_coords = findall(x -> x[2] ≈ 0.0, eachcol(coords))
  right_coords = findall(x -> x[1] ≈ 1.0, eachcol(coords))
  left_coords = findall(x -> x[1] ≈ 0.0, eachcol(coords))

  nsets = [
    NodeSet(1, top_coords)
    NodeSet(2, bottom_coords)
    NodeSet(3, right_coords)
    NodeSet(4, left_coords)
  ]

  ssets = []

  return Mesh{Int64, Float64}(coords, blocks, nsets, ssets)
end
