struct Mesh{Rtype, I, B}
  coords::Matrix{Rtype}
  blocks::Vector{Block{I, B}}
  nsets::Vector{NodeSet{I, B}}
  ssets::Vector{SideSet{I, B}}
end

# for now we read in all blocks
# to make omitting blocks possible we'll
# need to create our own native ids and reorder things
# function Mesh(
#   file_name::String,
#   nsets::Vector{T} = T[],
#   ssets::Vector{T} = T[]
# ) where T <: Union{Int64, String}

function Mesh(
  file_name::String;
  nsets = [],
  ssets = []
)
  exo = ExodusDatabase(file_name, "r")
  _, I, B, F = Exodus.int_and_float_modes(exo.exo)
  coords = read_coordinates(exo)
  # blocks_read = Vector{Block{I, B}}(undef, length(blocks))
  blocks_read = read_sets(exo, Block)
  nsets_read  = Vector{NodeSet{I, B}}(undef, length(nsets))
  ssets_read  = Vector{SideSet{I, B}}(undef, length(ssets))

  # for (n, id) in enumerate(blocks)
  #   blocks_read[n] = Block(exo, id)
  # end

  for (n, id) in enumerate(nsets)
    nsets_read[n] = NodeSet(exo, id)
  end

  for (n, id) in enumerate(ssets)
    ssets_read[n] = SideSet(exo, id)
  end

  return Mesh{F, I, B}(coords, blocks_read, nsets_read, ssets_read)
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
  # conns = Vector{Int64}(undef, 2 * e_x * e_y)

  e = 1
  for i in 0:e_x - 1
    for j in 0:e_y - 1
      conns[1, e] = i + n_x * j
      conns[2, e] = i + 1 + n_x * j
      conns[3, e] = i + 1 + n_x * (j + 1)
      # conns[e] = [
      #   i + n_x * j
      #   i + 1 + n_x * j
      #   i + 1 + n_x * (j + 1)
      # ]
      #
      e = e + 1
      conns[1, e] = i + n_x * j
      conns[2, e] = i + 1 + n_x * (j + 1)
      conns[3, e] = i + n_x * (j + 1)
      # conns[e] = [
      #   i + n_x * j
      #   i + 1 + n_x * (j + 1)
      #   i + n_x * (j + 1)
      # ]
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

  return Mesh{Float64, Int64, Int64}(coords, blocks, nsets, ssets)
end