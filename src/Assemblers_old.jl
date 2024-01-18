abstract type AbstractAssembler{Rtype, Itype} end

struct StaticAssembler{
  Rtype, Itype, 
  V <: AbstractArray{<:Number, 1}, 
  S <: AbstractSparseMatrix
} <: AbstractAssembler{Rtype, Itype}
  R::V
  K::S
end

int_type(::StaticAssembler{R, I, V, S}) where {R, I, V, S} = I
float_type(::StaticAssembler{R, I, V, S}) where {R, I, V, S} = R

# TODO type int
function StaticAssembler(dof::DofManager, fspaces::Fs) where Fs
  IJs = Tuple{Int64, Int64}[]
  for fspace in fspaces
    for e in 1:num_elements(fspace)
      conn = dof_connectivity(fspace, e)
      for temp in Iterators.product(conn, conn)
        push!(IJs, temp)
      end
    end
  end
  IJs = unique!(IJs)
  Is  = map(x -> x[2], IJs)
  Js  = map(x -> x[1], IJs)
  Vs  = zeros(Float64, size(Is))
  R   = zeros(Float64, num_nodes(dof) * num_dofs_per_node(dof))
  K   = sparse(Is, Js, Vs)
  return StaticAssembler{Float64, Int64, typeof(R), typeof(K)}(R, K)
end

# assembly methods, need different ones for what we're doing

function assemble_residual!(
  R,
  R_el, conn
)
  for i in axes(conn, 1)
    R[conn[i]] += R_el[i]
  end
end

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
) where M1 <: AbstractMatrix{<:Number}
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