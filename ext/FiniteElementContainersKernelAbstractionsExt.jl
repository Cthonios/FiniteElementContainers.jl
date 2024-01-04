module FiniteElementContainersKernelAbstractionsExt

using FiniteElementContainers
using KernelAbstractions

# DofManagers
function FiniteElementContainers.create_fields(
  backend::Backend, 
  ::DofManager{T, N, ND, NN, B}, 
  float_type::Type{<:Number} = Float64
) where {T, N, ND, NN, B <: AbstractArray{Bool, 1}}
  vals = KernelAbstractions.zeros(backend, float_type, ND * NN)
  return NodalField{ND, NN, Vector}(vals)
end

function FiniteElementContainers.create_fields(
  backend::Backend, 
  ::DofManager{T, N, ND, NN, B}, 
  float_type::Type{<:Number} = Float64
) where {T, N, ND, NN, B <: AbstractArray{Bool, 2}}
  vals = KernelAbstractions.zeros(backend, float_type, ND, NN)
  return NodalField{ND, NN, Matrix}(vals)
end

function FiniteElementContainers.create_unknowns(
  backend::Backend,
  dof::DofManager{T, N, ND, NN, Bools}, 
  float_type::Type{<:Number} = Float64
) where {T, N, ND, NN, Bools}
  return KernelAbstractions.zeros(backend, float_type, length(dof.unknown_indices))
end

@kernel function update_fields_kernel!(U::NodalField, dof::DofManager, Uu::V) where V <: AbstractVector
  I = @index(Global)
  U[dof.unknown_indices[I]] = Uu[I]
end

# TODO add wg size as parameter
function FiniteElementContainers.update_fields!(backend::Backend, U::NodalField, dof::DofManager, Uu::V) where V <: AbstractVector
  @assert length(Uu) == sum(dof.is_unknown)
  kernel = update_fields_kernel!(backend)
  kernel(U, dof, Uu, ndrange=length(Uu))
  synchronize(backend)
end

@kernel function update_unknown_ids_nset_kernel!(dof_manager, nset, dof)
  I = @index(Global)
  dof_manager[dof, nset[I]] = 0
end

@kernel function update_unknown_ids_kernel!(dof, ids)
  I = @index(Global)
  dof.unknown_indices[I] = ids[dof.unknown_indices[I]]
end

function FiniteElementContainers.update_unknown_ids!(
  backend::Backend,
  dof::DofManager,
  nsets::Nsets,
  dofs::Dofs
) where {Nsets, Dofs}

  kernel1 = update_unknown_ids_nset_kernel!(backend)
  kernel2 = update_unknown_ids_kernel!(backend)
  @assert length(nsets) == length(dofs)
  for (n, nset) in enumerate(nsets)
    @assert dofs[n] > 0 && dofs[n] <= num_dofs_per_node(dof)
    kernel1(dof, nset, dofs[n], ndrange=length(nset))
  end
  resize!(dof.unknown_indices, sum(dof.is_unknown))
  ids = FiniteElementContainers.dof_ids(dof)
  kernel2(dof, ids, ndrange=length(dof.unknown_indices))
end

end # module