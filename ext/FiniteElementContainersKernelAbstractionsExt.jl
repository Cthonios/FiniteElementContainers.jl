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

# Not sure how to do this one....
# @kernel function update_unknown_ids_kernel!(
#   dof_manager::DofManager, nodes::Nodes, dof::Int
# ) where Nodes
#   I = @index(Global)
#   @time @views dof_manager[dof, nodes[I]] = 0
#   # @time dof_manager[dof, I] = 0
# end

# function FiniteElementContainers.update_unknown_ids!(
#   backend::Backend, dof_manager::DofManager, nsets::NSets, dofs::Dofs
# ) where {NSets, Dofs}

#   @assert length(nsets) == length(dofs)
#   kernel = update_unknown_ids_kernel!(backend)
#   for (n, nset) in enumerate(nsets)
#     @assert dofs[n] > 0 && dofs[n] <= num_dofs_per_node(dof_manager)
#     kernel(dof_manager, nset, dofs[n], ndrange=length(nset))
#     synchronize(backend)
#   end
#   resize!(dof_manager.unknown_indices, sum(dof_manager.is_unknown))
#   ids = FiniteElementContainers.dof_ids(dof_manager)
#   dof_manager.unknown_indices .= ids[dof_manager.is_unknown]
# end

# function FiniteElementContainers.element_level_fields(
#   backend::Backend,
#   fspace::Fini
# )

end # module