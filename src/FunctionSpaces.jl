element_types = Dict{String, Type{<:ReferenceFiniteElements.ReferenceFEType}}(
  "HEX8"  => Hex8,
  "QUAD4" => Quad4,
  "QUAD9" => Quad9,
  "TET4"  => Tet4,
  "TET10" => Tet10,
  "TRI3"  => Tri3,
  "TRI6"  => Tri6
)

# main struct for a functionspace
# alternatives may be necessary e.g. if you want hessians or no gradients, etc.
struct QuadraturePointInterpolants{N, D, Rtype, L}
  X::SVector{D, Rtype}
  N::SVector{N, Rtype}
  ∇N_X::SMatrix{N, D, Rtype, L}
  JxW::Rtype
end

@kernel function setup_function_space_kernel!(
  fspace::S1,
  el_coords::S2,
  re::R
) where {S1, S2, R}

  q, e  = @index(Global, NTuple)
  X     = el_coords[e]
  N     = shape_function_values(re, q)
  ∇N_ξ  = shape_function_gradients(re, q)
  J     = X * ∇N_ξ
  J_inv = inv(J)
  ∇N_X  = (J_inv * shape_function_gradients(re, q)')'
  JxW   = det(J) * quadrature_weight(re, q)

  # set them in the structarray
  fspace.X[q, e]    = X * N
  fspace.N[q, e]    = N
  fspace.∇N_X[q, e] = ∇N_X
  fspace.JxW[q, e]  = JxW
end

struct FunctionSpace{S, C} #<: AbstractArray
  fspace::S
  connectivity::C
end

Base.getindex(f::F, q::Int, e::Int) where F <: FunctionSpace = f.fspace[q, e]
Base.size(f::F) where F <: FunctionSpace = size(f.fspace)
Base.axes(f::F, n::Int) where F <: FunctionSpace = Base.OneTo(size(f.fspace, n)) 

connectivity(f::F) where F <: FunctionSpace = f.connectivity

# CPU implementation, others in extensions
function FunctionSpace(
  coords::M1,
  block::MeshBlock{I, M2},
  re,
  n_dofs::Int,
  dev::CPU,
  wg_size::Int
) where {M1 <: AbstractMatrix, I, M2 <: AbstractMatrix}

  # extract some types
  Rtype = eltype(coords)
  D = eltype(re.interpolants).parameters[2]
  N = eltype(re.interpolants).parameters[1]

  # setup up arrays for dispatch to kernels
  n_nodes   = size(coords, 2)
  n_els     = size(block.conn, 2)
  n_qs      = length(re.interpolants.N)
  el_coords = StructArray(@views reinterpret(SMatrix{D, N, Rtype, D * N}, vec(coords[:, block.conn])))
  fspace    = StructArray{QuadraturePointInterpolants{N, D, Rtype, N * D}}(undef, n_qs, n_els)

  # create kernels
  setup_kernel = setup_function_space_kernel!(dev, wg_size)

  # dispatch kernels
  setup_kernel(fspace, el_coords, re, ndrange=(n_qs, n_els))

  # connectivity for this function space
  conn = Connectivity(block, n_nodes, n_dofs)

  # return fspace

  return FunctionSpace(fspace, conn)
end

function FunctionSpace(
  mesh::Mesh{M, V1, V2, V3},
  block_id::Int,
  q_degree::Int,
  n_dofs::Int,
  dev::Backend = CPU(),
  wg_size::Int = 1
) where {M, V1, V2, V3}

  # blocks_match = findall(x -> x.id == block_id, mesh.blocks)
  block_index = findfirst(x -> x.id == block_id, mesh.blocks)

  coords = mesh.coords
  block  = mesh.blocks[block_index]

  # get types
  Itype = eltype(connectivity(block))
  Rtype = eltype(coords)

  # setup reference element
  el = element_types[mesh.el_types[block_index]](q_degree)
  re = ReferenceFE(el, Itype, Rtype)

  # now call the actual local method
  FunctionSpace(coords, block, re, n_dofs, dev, wg_size)
end