struct EssentialBC{Itype, D, Rtype}
  nodes::Vector{Itype}
  coords::Vector{SVector{D, Rtype}}
  dof::Int
end

function EssentialBC(mesh::Mesh, nset_id::Int, dof::Int; Itype::Type = Int64, Rtype::Type = Float64)
  nset = filter(x -> x.id == nset_id, mesh.nsets)
  if length(nset) < 0
    throw(BoundsError("nset id: $nset_id not found"))
  else
    nset = nset[1]
  end
  D = size(mesh.coords, 1)
  coords = @views reinterpret(SVector{D, Rtype}, vec(mesh.coords[:, nset.nodes]))
  return EssentialBC{Itype, D, Rtype}(nset.nodes, coords, dof)
end
