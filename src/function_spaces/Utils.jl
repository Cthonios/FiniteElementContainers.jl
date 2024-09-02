elem_type_map = Dict{String, Type{<:ReferenceFiniteElements.ReferenceFEType}}(
  "HEX"     => Hex8,
  "HEX8"    => Hex8,
  "QUAD"    => Quad4,
  "QUAD4"   => Quad4,
  "QUAD9"   => Quad9,
  "TRI"     => Tri3,
  "TRI3"    => Tri3,
  "TRI6"    => Tri6,
  "TET"     => Tet4,
  "TETRA4"  => Tet4,
  "TETRA10" => Tet10
)

#####################################################

function volume(X, ∇N_ξ)
  J = (X * ∇N_ξ)'
  return det(J)
end

function map_shape_function_gradients(X, ∇N_ξ)
  J     = (X * ∇N_ξ)'
  J_inv = inv(J)
  ∇N_X  = (J_inv * ∇N_ξ')'
  return ∇N_X
end

# TODO should we deprecate these?
function setup_reference_element(
  type::Type{<:ReferenceFiniteElements.ReferenceFEType}, 
  q_degree
)
  ReferenceFiniteElements.ReferenceFE(type(Val(q_degree)))
end

function setup_dof_connectivity!(dof_conns, ids, conns)
  for e in 1:num_elements(dof_conns)
    conn            = connectivity(conns, e)
    # dof_conns[:, e] = ids[:, vec(conn)]
    dof_conns[:, e] = ids[:, conn]
  end
end
