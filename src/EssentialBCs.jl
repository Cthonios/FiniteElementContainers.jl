struct EssentialBC{Itype, Rtype}
  nodes::Vector{Itype}
  # coords::Vector{SVector{D, Rtype}}
  dof::Int
end

function EssentialBC(mesh::Mesh{F, I, B}, id::Int, dof::Int) where {F, I, B}
  nset = mesh.nsets[id]
  # D = size(mesh.coords, 1)
  # coords = @views reinterpret(SVector{D, Float64}, vec(mesh.coords[:, nset.nodes]))
  # return EssentialBC{Int64, D, Float64}(nset.nodes, coords, dof)
  # return EssentialBC{B, D, Float64}(nset.nodes, coords, dof)
  return EssentialBC{B, Float64}(nset.nodes, dof)
end 
