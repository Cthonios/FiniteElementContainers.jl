module FiniteElementContainersKernelAbstractionsExt

using FiniteElementContainers
using KernelAbstractions

# DofManagers
function FiniteElementContainers.create_fields(
  ::DofManager{T, ND, NN, A, V},
  backend::Backend
) where {T, ND, NN, A <: AbstractVector, V}
  vals = KernelAbstractions.zeros(backend, eltype(A), ND * NN)
  return NodalField{ND, NN, Vector}(vals)
end

function FiniteElementContainers.create_fields(
  ::DofManager{T, ND, NN, A, V},
  backend::Backend
) where {T, ND, NN, A <: AbstractMatrix, V}
  vals = KernelAbstractions.zeros(backend, eltype(A), ND, NN)
  return NodalField{ND, NN, Matrix}(vals)
end

function FiniteElementContainers.create_unknowns(
  dof::DofManager{T, ND, NN, A, V},
  backend::Backend
) where {T, ND, NN, A, V}
  return KernelAbstractions.zeros(backend, eltype(A), length(dof.unknown_dofs))
end

@kernel function update_fields_kernel!(U::NodalField, dof::DofManager, Uu::V) where V <: AbstractVector
  I = @index(Global)
  U[dof.unknown_dofs[I]] = Uu[I]
end

# TODO add wg size as parameter
function FiniteElementContainers.update_fields!(U::NodalField, dof::DofManager, Uu::V, backend::Backend) where V <: AbstractVector
  @assert length(Uu) == length(dof.unknown_dofs)#sum(dof.is_unknown)
  kernel = update_fields_kernel!(backend)
  kernel(U, dof, Uu, ndrange=length(Uu))
  synchronize(backend)
end

@kernel function update_unknown_dofs_kernel!(dof, dofs)
  I = @index(Global)
  dof.unknown_dofs[I] = dofs[I]
  nothing
end

function FiniteElementContainers.update_unknown_dofs!(
  dof::DofManager{T, ND, NN, A, V},
  backend::Backend 
) where {T, ND, NN, A, V}
  resize!(dof.unknown_dofs, ND * NN)
  dofs = FiniteElementContainers.dof_ids(dof)
  kernel = update_unknown_dofs_kernel!(backend)
  kernel(dof, dofs, ndrange=length(dofs))
  nothing
end

# function FiniteElementContainers.update_unknown_dofs!(
#   dof::DofManager{T, ND, NN, A, V1},
#   dofs::V2,
#   backend::Backend 
# ) where {T, ND, NN, A, V1, V2 <: AbstractVector{<:Integer}}
#   resize!(dof.unknown_dofs, ND * NN)
#   dofs_all = FiniteElementContainers.dof_ids(dof)
#   kernel = update_unknown_dofs_kernel!(backend)
#   kernel(dof, dofs_all, ndrange=length(dofs_all))
#   # setdiff!(dofs_all, dofs)
#   # resize!(dof.unknown_indices, length(dofs_all))
#   # dof.unknown_indices .= dofs_all
#   # deleteat!(dof.unknown_dofs, dofs)
#   setdiff!(dof.unknown_dofs, dofs)
#   nothing
# end

# @kernel function update_unknown_dofs_non_default_kernel!(dof, dofs)
#   I
# end
# @kernel function update_unknown_ids_nset_kernel!(dof_manager, nset, dof)
#   I = @index(Global)
#   dof_manager[dof, nset[I]] = 0
# end

# @kernel function update_unknown_ids_kernel!(dof, ids)
#   I = @index(Global)
#   dof.unknown_indices[I] = ids[dof.unknown_dofs[I]]
# end

# function FiniteElementContainers.update_unknown_dofs!(
#   dof::DofManager,
#   nsets::Nsets,
#   dofs::Dofs,
#   backend::Backend
# ) where {Nsets, Dofs}

#   kernel1 = update_unknown_ids_nset_kernel!(backend)
#   kernel2 = update_unknown_ids_kernel!(backend)
#   @assert length(nsets) == length(dofs)
#   for (n, nset) in enumerate(nsets)
#     @assert dofs[n] > 0 && dofs[n] <= num_dofs_per_node(dof)
#     kernel1(dof, nset, dofs[n], ndrange=length(nset))
#   end
#   resize!(dof.unknown_indices, length(dof.unknown_dofs))
#   ids = FiniteElementContainers.dof_ids(dof)
#   kernel2(dof, ids, ndrange=length(dof.unknown_dofs))
#   synchronize(backend)
# end

end # module