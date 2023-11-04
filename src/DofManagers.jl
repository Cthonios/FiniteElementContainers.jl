struct DofManager{NDOFS, B <: AbstractMatrix{Bool}, V <: AbstractArray{<:Integer}, Rtype <: AbstractFloat}
  is_unknown::B
  unknown_indices::V
end

function DofManager(
  mesh::M, n_dofs::Int; 
  Rtype::Type{<:AbstractFloat} = Float64
) where M <: Mesh
  n_nodes = size(mesh.coords, 2)

  is_unknown      = BitArray(1 for _ = 1:n_dofs, _ = 1:n_nodes)
  ids             = reshape(1:n_nodes * n_dofs, n_dofs, n_nodes)
  unknown_indices = ids[is_unknown]

  B = typeof(is_unknown)
  V = typeof(unknown_indices)

  return DofManager{n_dofs, B, V, Rtype}(is_unknown, unknown_indices)
end

Base.size(dof::DofManager) = size(dof.is_unknown)
KernelAbstractions.get_backend(dof::DofManager) = get_backend(dof.unknown_indices)

ids(d::DofManager) = reshape(1:length(d.is_unknown), size(d.is_unknown))
n_dofs(dof::DofManager)  = size(dof.is_unknown, 1)
n_nodes(dof::DofManager) = size(dof.is_unknown, 2)
fieldtype(::DofManager{NDOFS, B, V, Rtype}) where {NDOFS, B, V, Rtype} = Rtype

create_fields(dof::DofManager)   = KernelAbstractions.zeros(get_backend(dof), fieldtype(dof), size(dof))
create_unknowns(dof::DofManager) = KernelAbstractions.zeros(get_backend(dof), fieldtype(dof), sum(dof.is_unknown))

# function update_bcs!(U::S, )

@kernel function update_bc_kernel!(U, mesh_coords, dof, bc)
  node = @index(Global)
  dof.is_unknown[bc.dof, node] = 0
  U[bc.dof, node] = @views bc.func(mesh_coords[:, node], 0.)
end

function update_bcs_old!(
  U::Mat, mesh::M, dof::DofManager, bcs::V
) where {Mat <: AbstractMatrix, M <: Mesh, V <: Vector{<:EssentialBC}}

  dof.is_unknown .= 1
  for bc in bcs
    for node in bc.nodes
      dof.is_unknown[bc.dof, node] = 0
      U[bc.dof, node] = @views bc.func(mesh.coords[:, node], 0.)
    end
  end

  # TODO below line is the only source of allocations here
  new_unknown_indices = @views ids(dof)[dof.is_unknown]
	resize!(dof.unknown_indices, length(new_unknown_indices))
	dof.unknown_indices .= new_unknown_indices
end


# function update_bcs!(
#   U::Mat, mesh::M, dof::DofManager, bcs::V, 
#   dev::Backend, wg_size::Int
# ) where {Mat <: AbstractMatrix, M <: Mesh, V <: Vector{<:EssentialBC}}

#   kernel! = update_bc_kernel!(dev, wg_size)
#   dof.is_unknown .= 1
#   for bc in bcs
#     kernel!(U, dof, mesh, bc, ndrange=length(bc.nodes))
#   end

#   # TODO below line is the only source of allocations here
#   new_unknown_indices = @views ids(dof)[dof.is_unknown]
# 	resize!(dof.unknown_indices, length(new_unknown_indices))
# 	dof.unknown_indices .= new_unknown_indices
# end

function update_bcs!(U::Mat, mesh::M, dof::DofManager, bcs::V) where {Mat <: AbstractMatrix, M <: Mesh, V <: Vector{<:EssentialBC}}
  kernel! = update_bc_kernel!(get_backend(U))

  dof.is_unknown .= 1
  for bc in bcs
    @time kernel!(U, mesh.coords, dof, bc, ndrange=length(bc.nodes))
  end
end 


# TODO adding @Const to Uu adds some serious allocations.
# Find out the right way to parametrically type @Consts
@kernel function update_fields_kernel!(U::M, @Const(dof::DofManager), Uu::V) where {M <: AbstractMatrix, V <: AbstractVector}
  n = @index(Global)
  i = dof.unknown_indices[n]
  U[i] = Uu[n]
end

function update_fields!(U::M, dof::DofManager, Uu::V) where {M <: AbstractMatrix, V <: AbstractVector}
  kernel! = update_fields_kernel!(get_backend(U))
  kernel!(U, dof, Uu, ndrange=length(Uu))
end