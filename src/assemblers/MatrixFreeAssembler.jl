
struct MatrixFreeAssembler{Rtype, Itype} <: Assembler{Rtype, Itype}
end

"""
assembly method for matrix free assembler
when functions for the element level residual and tangent_func
are provided
"""
function assemble!(
  assembler::MatrixFreeAssembler,
  dof::DofManager,
  fspace::FunctionSpace,
  X, U, V,
  residual_func, tangent_func
)

  NDof = num_dofs_per_node(dof)
  N    = num_nodes_per_element(fspace)
  NxNDof = N * NDof

  for e in 1:num_elements(fspace)
    U_el = element_level_fields(fspace, U, e)
    V_el = element_level_fields(fspace, V, e)
    R_el = zeros(SVector{NxNDof, Float64})
    Hv_el = zeros(SVector{NxNDof, Float64})

    # quadrature loop
    for q in 1:num_q_points(fspace)
      fspace_values = getindex(fspace, X, q, e)
      R_el = R_el + residual_func(fspace_values, U_el)
      Hv_el = Hv_el + dot(tangent_func(fspace_values, U_el), V_el)
    end

    # assemble residual using connectivity here
    conn = dof_connectivity(fspace, e)
    assemble!(assembler, R_el, conn)
    assemble!(assembler, Hv_el, conn)
  end
end