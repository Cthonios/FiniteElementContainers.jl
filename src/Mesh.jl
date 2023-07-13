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