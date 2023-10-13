module FiniteElementContainersAdaptExt

using Adapt
using CUDA
using CUDA.CUDAKernels
using Exodus
using FiniteElementContainers
using StaticArrays
using StructArrays

# Exodus types - eventually maybe move to exodus?
function Adapt.adapt_structure(to, block::FiniteElementContainers.MeshBlock{I, B}) where {I, B}
  id   = Adapt.adapt_structure(to, block.id)
  conn = Adapt.adapt_structure(to, block.conn)
  return FiniteElementContainers.MeshBlock{I, B}(id, conn)
end

function Adapt.adapt_structure(to, nset::NodeSet{I, B}) where {I, B}
  id    = Adapt.adapt_structure(to, nset.id)
  nodes = Adapt.adapt_structure(to, nset.nodes)
  return NodeSet{I, B}(id, nodes)
end

function Adapt.adapt_structure(to, sset::SideSet{I, B}) where {I, B}
  id       = Adapt.adapt_structure(to, sset.id)
  elements = Adapt.adapt_structure(to, sset.elements)
  sides    = Adapt.adapt_structure(to, sset.sides)
  return SideSet{I, B}(id, elements, sides)
end

# Mesh type
# function Adapt.adapt_structure(to, mesh::Mesh{F, I, B}) where {F, I, B}
#   coords = Adapt.adapt_structure(to, mesh.coords)
#   blocks = Adapt.adapt_structure.(to, mesh.blocks) |> to
#   nsets  = Adapt.adapt_structure.(to, mesh.nsets) |> to
#   ssets  = Adapt.adapt_structure.(to, mesh.ssets) |> to
#   return Mesh(coords, blocks, nsets, ssets)
# end 

# Element level coordinates
# function Adapt.adapt_structure(to, e::FiniteElementContainers.ElementLevelCoordinates{T}) where T
#   el_coords = Adapt.adapt_structure(to, e.el_coords)
#   return FiniteElementContainers.ElementLevelCoordinates(el_coords)
# end

# MOVE TO DIFFERENT EXT EVENTUALLY
# FunctionSpace for GPU
function FiniteElementContainers.FunctionSpace(
  coords::M1,
  block::FiniteElementContainers.MeshBlock{I, M2},
  re,
  dev::CUDABackend,
  wg_size::Int
) where {M1 <: AbstractMatrix, I, M2 <: AbstractMatrix}

  # extract some types
  Rtype = eltype(coords)
  D = eltype(re.interpolants).parameters[2]
  N = eltype(re.interpolants).parameters[1]

  # setup up arrays for dispatch to kernels
  n_els     = size(block.conn, 2)
  n_qs      = length(re.interpolants.N)
  el_coords = StructArray(@views reinterpret(SMatrix{D, N, Rtype, D * N}, vec(coords[:, block.conn])))
  fspace    = StructArray{FiniteElementContainers.QuadraturePointInterpolants{N, D, Rtype, N * D}}(undef, n_qs, n_els)

  # move things to devices
  fspace_dev    = replace_storage(CuArray, fspace)
  el_coords_dev = replace_storage(CuArray, el_coords)
  re_dev        = Adapt.adapt_structure(CuArray, re)

  # create kernels
  setup_kernel = FiniteElementContainers.setup_function_space_kernel!(dev, wg_size)

  # dispatch kernels
  setup_kernel(fspace_dev, el_coords_dev, re_dev, ndrange=(n_qs, n_els))

  return fspace_dev
end

# boundary conditions below

function Adapt.adapt_structure(to, bc::EssentialBC{V, F}) where {V, F}
  nodes = Adapt.adapt_structure(to, bc.nodes)
  dof   = bc.dof
  func  = bc.func

  return EssentialBC(nodes, dof, func)
end

end