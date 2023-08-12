struct EssentialBC{Itype, D, Rtype}
  nodes::Vector{Itype}
  coords::Vector{SVector{D, Rtype}}
  dof::Int
end

function EssentialBC(mesh::Mesh, id::Int, dof::Int)
  nset = mesh.nsets[id]
  D = size(mesh.coords, 1)
  coords = @views reinterpret(SVector{D, Float64}, vec(mesh.coords[:, nset.nodes]))
  return EssentialBC{Int64, D, Float64}(nset.nodes, coords, dof)
end 
