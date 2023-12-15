abstract type AbstractAssembler{Rtype, Itype} end

function setup_hessian_coordinates!(
  row_coords::Is, col_coords::Js,
  fspaces::Fs
) where {Is <: AbstractArray{<:Integer, 1}, 
         Js <: AbstractArray{<:Integer, 1},
         Fs}

  counter = 1
  # need below to really be "dof connectivity"
  for fspace in fspaces
    # conn = dof_connectivity(dof, fspace)
    conn = dof_connectivity(fspace)
    for e in axes(conn, 2)
      for i in axes(conn, 1)
        for j in axes(conn, 1)
          row_coords[counter] = conn[i, e]
          col_coords[counter] = conn[j, e]
          counter += 1
        end
      end
    end
  end
end

struct StaticAssembler{
  Rtype, Itype, 
  V <: AbstractArray{<:Number, 1}, 
  S <: AbstractSparseMatrix
} <: AbstractAssembler{Rtype, Itype}
  R::V
  K::S
end

function StaticAssembler(dof::DofManager, fspaces::Fs) where Fs #<: AbstractArray{<:FunctionSpace, 1}
  n_hessian_entries = 0
  # TODO add functionality to only size things based on nodes
  # seen in function space connectivity
  # uniques = Vector{Int64}(undef, 0)
  for fspace in fspaces
    # conn = dof_connectivity(dof, fspace)
    conn = dof_connectivity(fspace)
    n_hessian_entries += size(conn, 2) * (size(conn, 1)^2)

    # append!(uniques, unique(conn))
  end
  # uniques = unique(uniques)

  Is = Vector{Int64}(undef, n_hessian_entries)
  Js = Vector{Int64}(undef, n_hessian_entries)
  Vs = zeros(Float64, n_hessian_entries)

  setup_hessian_coordinates!(Is, Js, fspaces)

  R = zeros(Float64, num_nodes(dof) * num_dofs_per_node(dof))
  K = sparse(Is, Js, Vs)
  return StaticAssembler{Float64, Int64, typeof(R), typeof(K)}(R, K)
end

# assembly methods, need different ones for what we're doing

function assemble!(
  R::V1,
  R_el, conn
) where V1 <: AbstractVector
  for i in axes(conn, 1)
    R[conn[i]] += R_el[i]
  end
end

function assemble!(
  assembler::StaticAssembler,
  R_el::V, conn
) where V <: AbstractVector
  for i in axes(conn, 1)
    assembler.R[conn[i]] += R_el[i]
  end
end

function assemble!(
  K::M1,
  K_el, conn
) where M1 <: AbstractMatrix
  for i in axes(conn, 1)
    # assembler.R[conn[i]] += R_el[i]
    for j in axes(conn, 1)
      K[conn[i], conn[j]] += K_el[i, j]
    end
  end
end

function assemble!(
  assembler::StaticAssembler,
  R_el::V, K_el::M, conn
) where {V <: AbstractVector, M <: AbstractMatrix}
  for i in axes(conn, 1)
    assembler.R[conn[i]] += R_el[i]
    for j in axes(conn, 1)
      assembler.K[conn[i], conn[j]] += K_el[i, j]
    end
  end
end

function assemble!(
  assembler::StaticAssembler,
  dof::DofManager,
  fspace::FunctionSpace,
  residual_func::F1,
  tangent_func::F2,
  X::V1,
  U::V2
) where {F1 <: Function, F2 <: Function, 
         V1 <: AbstractArray, V2 <: AbstractArray}

  NDof   = num_dofs_per_node(dof)
  N      = num_nodes_per_element(fspace)
  NxNDof = N * NDof

  for e in 1:num_elements(fspace)
    # U_el = element_level_fields(fspace, e, U)
    U_el = element_level_fields(fspace, U, e)
    R_el = zeros(SVector{NxNDof, Float64})
    K_el = zeros(SMatrix{NxNDof, NxNDof, Float64, NxNDof * NxNDof})
    for q in num_q_points(fspace)
      fspace_values = getindex(fspace, X, q, e)
      R_el = R_el + residual_func(fspace_values, U_el)
      K_el = K_el + tangent_func(fspace_values, U_el)
    end
    # @show U_el
    # @show K_el
    # allocations below
    # conn = dof_connectivity(dof, fspace, e)
    conn = dof_connectivity(fspace, e)
    assemble!(assembler, R_el, K_el, conn)
  end
end

function assemble!(
  assembler::StaticAssembler,
  dof::DofManager,
  fspaces::Fs,
  residual_func::F1,
  tangent_func::F2,
  X::V1,
  U::V2
) where {Fs <: AbstractArray{<:FunctionSpace}, 
         F1 <: Function, F2 <: Function, 
         V1 <: AbstractArray, V2 <: AbstractArray}

  assembler.R .= 0.
  assembler.K .= 0.

  for fspace in fspaces
    assemble!(assembler, dof, fspace, residual_func, tangent_func, X, U)
  end 

end

function remove_constraints(asm::StaticAssembler, dof::DofManager)
  R = asm.R[dof.unknown_indices]
  K = asm.K[dof.unknown_indices, dof.unknown_indices]
  return R, K
end