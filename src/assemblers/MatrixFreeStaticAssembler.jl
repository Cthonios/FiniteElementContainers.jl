
struct MatrixFreeStaticAssembler{
  Rtype, Itype,
  R, Kv
} <: Assembler{Rtype, Itype}
  residuals::R
  stiffness_actions::Kv
end

function MatrixFreeStaticAssembler(dof::DofManager)
  residuals = create_fields(dof)
  stiffness_actions = create_fields(dof)
  return MatrixFreeStaticAssembler{Float64, Int64, typeof(residuals), typeof(stiffness_actions)}(
    residuals, stiffness_actions
  )
end

function assemble!(
  assembler::MatrixFreeStaticAssembler,
  R_el::V1, Kv_el::V1, conn::V2
) where {V1 <: AbstractVector{<:Number}, V2 <: AbstractVector{<:Integer}}

  for i in axes(conn, 1)
    assembler.residuals[conn[i]] += R_el[i]
    assembler.stiffness_actions[conn[i]] += Kv_el[i]
  end
  return nothing
end

"""
assembly method for matrix free assembler
when functions for the element level residual and tangent_func
are provided
"""
function assemble!(
  assembler::MatrixFreeStaticAssembler,
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
    Kv_el = zeros(SVector{NxNDof, Float64})

    # quadrature loop
    for q in 1:num_q_points(fspace)
      fspace_values = getindex(fspace, X, q, e)
      R_el = R_el + residual_func(fspace_values, U_el)
      Kv_el = Kv_el + dot(tangent_func(fspace_values, U_el), V_el)
    end

    # assemble residual using connectivity here
    conn = dof_connectivity(fspace, e)
    assemble!(assembler.residuals, R_el, conn)
    assemble!(assembler.stiffness_actions, Kv_el, conn)
  end
end