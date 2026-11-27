struct BlockColoring{IT <: Integer, IV <: AbstractVector{IT}}
  colors::IV
  # color_offsets::IV
  color_offsets::Vector{IT}
  color_elems::IV
end

num_colors(bc::BlockColoring) = length(bc.color_offsets) - 1

# currently not GPU safe
function get_elements_by_color(bc::BlockColoring, c::Integer)
  @boundscheck 1 <= c <= num_colors(bc) || throw(BoundsError(bc, c))
  return @view bc.color_elems[bc.color_offsets[c]:bc.color_offsets[c + 1] - 1]
end

# stuff for main struct setup below.
# current design is as follows
# MeshColoring is created off the mesh and the FunctionSpace will
# store a Vector of BlockColoring ordered in a way consistent with
# other block ordered containers

# needs to run off of mesh connectivity, not FunctionSpace connectivity
"""
Turn a per-element color vector into the CSR layout (counting sort, stable so
elements stay in increasing id order within a color, which helps memory
locality).
"""
function _build_block_coloring(colors::Vector{IT}, ncolors::Int) where IT
  counts = zeros(IT, ncolors)
  for c in colors
    counts[c] += one(IT)
  end

  offsets = Vector{IT}(undef, ncolors + 1)
  offsets[1] = one(IT)
  for c in 1:ncolors
    offsets[c + 1] = offsets[c] + counts[c]
  end

  elems = Vector{IT}(undef, length(colors))
  next = copy(offsets)
  for e in eachindex(colors)
    c = colors[e]
    elems[next[c]] = IT(e)
    next[c] += one(IT)
  end

  return BlockColoring{IT, Vector{IT}}(colors, offsets, elems)
end

function _color_block!(node_masks::Vector{UInt64}, conn::AbstractMatrix{IT}) where IT
  nodes_per_elem, n_elems = size(conn)
  colors = Vector{IT}(undef, n_elems)
  max_color = 0

  for e in 1:n_elems
    used = UInt64(0)
    @inbounds for i in 1:nodes_per_elem
      used |= node_masks[conn[i, e]]
    end

    # lowest unset bit; ~used has a 1 where a color is free
    free = ~used
    if free == 0
      error("Mesh coloring needs more than $(_MAX_COLORS) colors. " *
            "Increase mask width in MeshColoring.jl (e.g. use a tuple of UInt64 words).")
    end
    c = trailing_zeros(free) + 1
    bit = UInt64(1) << (c - 1)

    @inbounds for i in 1:nodes_per_elem
      node_masks[conn[i, e]] |= bit
    end

    colors[e] = c
    max_color = max(max_color, c)
  end

  return colors, max_color
end

struct MeshColoring{IT <: Integer, IV <: AbstractVector{IT}}
  blocks::Vector{BlockColoring{IT, IV}}
  num_colors::Int
  shared::Bool
end

function MeshColoring(
    mesh::AbstractMesh;
    share_across_blocks::Bool = true
)
  IT = eltype(mesh.node_id_map)
  block_names = mesh.element_block_names

  # number of nodes = largest node index referenced by any connectivity
  n_nodes = 0
  for name in block_names
    n_nodes = max(n_nodes, Int(maximum(mesh.element_conns[name])))
  end

  node_masks = zeros(UInt64, n_nodes)

  blocks = BlockColoring{IT, Vector{IT}}[]
  total_colors = 0

  for name in block_names
    if !share_across_blocks
      fill!(node_masks, UInt64(0))
    end
    colors, nc = _color_block!(node_masks, mesh.element_conns[name])
    push!(blocks, _build_block_coloring(colors, nc))
    total_colors = max(total_colors, nc)
  end

  # Pad offsets so every block has the same number of colors; this lets
  # callers loop `for c in 1:num_colors(mc)` over every block without checks.
  for n in axes(block_names, 1)
    bc = blocks[n]
    nc = num_colors(bc)
    if nc < total_colors
      offsets = vcat(
        bc.color_offsets,
        fill(bc.color_offsets[end], total_colors - nc)
      )
      blocks[n] = BlockColoring{IT, Vector{IT}}(bc.colors, offsets, bc.color_elems)
    end
  end

  return MeshColoring{IT, Vector{IT}}(
    blocks, total_colors, share_across_blocks
  )
end

# currently not GPU safe
function color_sizes(mc::MeshColoring)
  sizes = zeros(Int, mc.num_colors)
  for bc in values(mc.blocks)
    for c in 1:num_colors(bc)
      sizes[c] += bc.color_offsets[c + 1] - bc.color_offsets[c]
    end
  end
  return sizes
end

function get_element_block_colors(coloring::MeshColoring, block::Integer)
    return coloring.blocks[block].colors
end

function get_elements_by_color(coloring::MeshColoring, block::Integer, c::Integer)
    return get_elements_by_color(coloring.blocks[block], c)
end

function validate_coloring(mesh::AbstractMesh, mc::MeshColoring; verbose::Bool = false)
  n_nodes = 0
  for name in mesh.element_block_names
    n_nodes = max(n_nodes, Int(maximum(mesh.element_conns[name])))
  end

  # touched[n] = true if node n has been used by an element of the color
  # currently being checked
  touched = falses(n_nodes)

  for c in 1:mc.num_colors
    fill!(touched, false)
    for (b, name) in enumerate(mesh.element_block_names)
      if !mc.shared
        fill!(touched, false)
      end
      conn = mesh.element_conns[name]
      for e in get_elements_by_color(mc.blocks[b], c)
        for i in axes(conn, 1)
          n = conn[i, e]
          if touched[n]
            verbose && @warn "Coloring conflict" color=c block=name element=e node=n
            return false
          end
          touched[n] = true
        end
      end
    end
  end

  return true
end
