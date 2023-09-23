# call_bc_func(e::EssentialBC{Itype}, X, t) where {Itype} = getfield(e, :func)(X, t) 

struct EssentialBCNodeSetIDException <: Exception
  nset_id::Int
end

Base.show(io::IO, e::EssentialBCNodeSetIDException) = 
print(io, "\nNodeset ID $(e.nset_id) not found in mesh.\n")

ebc_nset_id_exception(nset_id::Int) = throw(EssentialBCNodeSetIDException(nset_id))

struct EssentialBC{V <: AbstractArray{<:Integer}, F}
  nodes::V
  dof::Int
  func::F
end

function EssentialBC(
  mesh::Mesh{M, V1, V2, V3},
  id::Int,
  dof::Int,
  func::Function = (x, t) -> 0.
) where {M, V1, V2, V3}

  mesh_nset_id = findfirst(x -> x.id == id, mesh.nsets)
  if mesh_nset_id === nothing
    ebc_nset_id_exception(id)
  end
  nset = mesh.nsets[mesh_nset_id]
  return EssentialBC(nset.nodes, dof, func)
end