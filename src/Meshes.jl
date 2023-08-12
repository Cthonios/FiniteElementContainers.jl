struct Mesh
  coords::Matrix{Float64}
  blocks::Vector{Block{Int64, Int64}}
  nsets::Vector{NodeSet{Int64, Int64}}
  ssets::Vector{SideSet{Int64, Int64}}
end

function Mesh(
  file_name::String, 
  blocks::Vector{T};
  nsets::Vector{T} = T[],
  ssets::Vector{T} = T[]
) where T <: Union{Int64, String}

  exo = ExodusDatabase(file_name, "r")

  coords = read_coordinates(exo)
  blocks_read = Vector{Block{Int64, Int64}}(undef, length(blocks))
  nsets_read  = Vector{NodeSet{Int64, Int64}}(undef, length(nsets))
  ssets_read  = Vector{SideSet{Int64, Int64}}(undef, length(ssets))


  for (n, id) in enumerate(blocks)
    block = Block(exo, id)
    blocks_read[n] = Block{Int64, Int64}(block.id, block.num_elem, 
                                         block.num_nodes_per_elem, 
                                         block.elem_type, block.conn)
  end

  for (n, id) in enumerate(nsets)
    nset = NodeSet(exo, id)
    nsets_read[n] = NodeSet{Int64, Int64}(nset.id, nset.nodes)
  end

  for (n, id) in enumerate(ssets)
    sset = SideSet(exo, id)
    ssets_read[n] = SideSet{Int64, Int64}(sset.id, sset.elements, sset.sides)
  end

  return Mesh(coords, blocks_read, nsets_read, ssets_read)
end

function Mesh(n_x::Int, n_y::Int, x_extent::Vector{<:Real}, y_extent::Vector{<:Real})
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

  nsets = NodeSet[
    NodeSet(1, top_coords)
    NodeSet(2, bottom_coords)
    NodeSet(3, right_coords)
    NodeSet(4, left_coords)
  ]

  ssets = []

  return Mesh(coords, blocks, nsets, ssets)
end