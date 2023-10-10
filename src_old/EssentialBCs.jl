struct EssentialBC{Itype}
  nodes::Vector{Itype}
  dof::Int
  func::Function
end

function EssentialBC(mesh::Mesh{F, I, B}, id::Int, dof::Int, func::Function = (x, t) -> 0.) where {F, I, B}
  nset = mesh.nsets[id]
  return EssentialBC{B}(nset.nodes, dof, func)
end 

call_bc_func(e::EssentialBC{Itype}, X, t) where {Itype} = getfield(e, :func)(X, t) 
