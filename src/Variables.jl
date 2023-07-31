function setup_element_level_variable!(
  el_vars::Vector{SMatrix{D, N, Ftype, L}},
  nodal_vars::Matrix{<:AbstractFloat}, block::Block{B}
) where {N, D, Ftype, L, B}

  for e in eachindex(el_vars)
    el_vars[e] = SMatrix{D, N, Ftype, L}(view(nodal_vars, :, view(block.conn, :, e)))
  end 
end


struct ElementLevelNodalVariables{N, D, Nvar, Ftype, L1, L2}
  X_els::Vector{SMatrix{D, N, Ftype, L1}}
  u_els::Vector{SMatrix{Nvar, N, Ftype, L2}}
end

function ElementLevelNodalVariables(
  coords::Matrix{<:AbstractFloat},
  block::Block{B}, 
  ::ReferenceFE{N, D, L, Itype, Ftype},
  ::Val{n_dof}
) where {B, N, D, L, Itype, Ftype, n_dof}

  u_nodal_init = zeros(Ftype, n_dof, size(coords, 2))
  X_els = Vector{SMatrix{D, N, Ftype, L}}(undef, block.num_elem)
  u_els = Vector{SMatrix{n_dof, N, Ftype, n_dof * N}}(undef, block.num_elem)

  setup_element_level_variable!(X_els, coords, block)
  setup_element_level_variable!(u_els, u_nodal_init, block)

  return ElementLevelNodalVariables{N, D, n_dof, Ftype, N * D, N * n_dof}(
    X_els, u_els
  )
end

function ElementLevelNodalVariables(
  coords::Matrix{<:AbstractFloat},
  block::Block{B}, 
  re::ReferenceFE{N, D, L, Itype, Ftype},
  n_dof::Int
) where {B, N, D, L, Itype, Ftype}

  return ElementLevelNodalVariables(coords, block, re, Val(n_dof))
end

# struct QuadraturePointVariables
# end