# temp structure since Block from Exodus
# has a string in it
struct MeshBlock{I, M <: AbstractMatrix{<:Integer}}
  id::I
  conn::M
end

function MeshBlock(b::Block{I, B}) where {I, B}
  return MeshBlock(b.id, b.conn)
end

connectivity(m::MeshBlock) = getfield(m, :conn)#m.conn

####################################

struct Mesh{M <: AbstractMatrix{<:AbstractFloat},
            V1 <: AbstractVector{<:MeshBlock},
            V2 <: AbstractVector{<:NodeSet},
            V3 <: AbstractVector{<:SideSet}}
  coords::M
  blocks::V1
  nsets::V2
  ssets::V3
  el_types::Vector{String}
end

coordinates(m::Mesh) = m.coords
blocks(m::Mesh) = m.blocks
nodesets(m::Mesh) = m.nsets
sidesets(m::Mesh) = m.ssets

function Mesh(file_name::String; nsets::Vector{<:Integer} = Int[], ssets::Vector{<:Integer} = Int[])
  exo = ExodusDatabase(file_name, "r")
  _, I, B, F = Exodus.int_and_float_modes(exo.exo)

  nsets = convert(Vector{I}, nsets)
  ssets = convert(Vector{I}, ssets)

  coords = read_coordinates(exo)
  blocks_read = read_sets(exo, Block)
  blocks = MeshBlock.(blocks_read)
  nsets_read  = Vector{NodeSet{I, Vector{B}}}(undef, length(nsets))
  ssets_read  = Vector{SideSet{I, Vector{B}}}(undef, length(ssets))
  el_types    = map(x -> x.elem_type, blocks_read)

  for (n, id) in enumerate(nsets)
    nsets_read[n] = NodeSet(exo, id)
  end

  for (n, id) in enumerate(ssets)
    ssets_read[n] = SideSet(exo, id)
  end
  
  # return Mesh{F, I, B}(coords, blocks, nsets_read, ssets_read)
  return Mesh(coords, blocks, nsets_read, ssets_read, el_types)
end