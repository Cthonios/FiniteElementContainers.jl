module FiniteElementContainersAdaptExt

using Adapt
using CUDA
using CUDA.CUDAKernels
using Exodus
using FiniteElementContainers
using StaticArrays
using StructArrays

# Exodus types - eventually maybe move to exodus?
function Adapt.adapt_storage(to, block::FiniteElementContainers.MeshBlock{I, B}) where {I, B}
  id   = Adapt.adapt_storage(to, block.id)
  conn = Adapt.adapt_storage(to, block.conn)
  return FiniteElementContainers.MeshBlock{I, B}(id, conn)
end

# Mesh type
function Adapt.adapt_storage(to, mesh::M) where M <: FiniteElementContainers.Mesh
  println("here")
  coords = Adapt.adapt_storage(to, mesh.coords)
  blocks = replace_storage(to, mesh.blocks)
  nsets  = replace_storage(to, mesh.nsets)
  ssets  = replace_storage(to, mesh.ssets)
  return Mesh(coords, blocks, nsets, ssets)
end 

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

function Adapt.adapt_storage(to, fspace::F) where F <: FiniteElementContainers.FunctionSpace
  fspace_dev = Adapt.adapt_storage(to, fspace.fspace)
  return FiniteElementContainers.FunctionSpace(fspace_dev)
end
# boundary conditions below

function Adapt.adapt_storage(to, bc::EssentialBC{V, F}) where {V, F}
  FiniteElementContainers.EssentialBC(
    Adapt.adapt_storage(to, bc.nodes), bc.dof, bc.func
  )
end

function Adapt.adapt_storage(to, conn::C) where C <: FiniteElementContainers.Connectivity
  FiniteElementContainers.Connectivity(
    conn.n_elm,
    conn.n_nodes_per_elem,
    conn.n_dofs,
    Adapt.adapt_storage(to, conn.conn),
    Adapt.adapt_storage(to, conn.dof_conn)
  )
end

# function Adapt.adapt_storage(to, conn::C) where C <: 

# DofManagers
function Adapt.adapt_storage(to, dof::D) where D <: FiniteElementContainers.DofManager

  is_unknown = Adapt.adapt_storage(to, dof.is_unknown)
  unknown_indices = Adapt.adapt_storage(to, dof.unknown_indices)
  B = typeof(is_unknown)
  V = typeof(unknown_indices)
  Rtype = FiniteElementContainers.fieldtype(dof)
  DofManager{B, V, Rtype}(
    is_unknown, unknown_indices
  )
  # DofManager(
  #   Adapt.adapt_storage(to, dof.is_unknown),
  #   Adapt.adapt_storage(to, dof.unknown_indices)
  # )
end

end # module