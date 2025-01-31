"""
$(TYPEDEF)
"""
abstract type Assembler{Rtype, Itype} end

"""
$(TYPEDSIGNATURES)
"""
int_type(::Assembler{R, I}) where {R, I} = I

"""
$(TYPEDSIGNATURES)
"""
float_type(::Assembler{R, I}) where {R, I} = R

"""
$(TYPEDSIGNATURES)
Assembly method for a scalar field stored as a size 1 vector
"""
function assemble!(
  global_val::Vector, 
  fspace, block_num, e, local_val
)
  global_val[1] += local_val
  return nothing
end

"""
$(TYPEDSIGNATURES)
generic assembly method that directly goes into a vector
"""
function assemble!(
  R::V1, R_el::V2, conn::V3
) where {V1 <: AbstractArray{<:Number}, 
         V2 <: AbstractArray{<:Number},
         V3 <: AbstractArray{<:Integer}}

  for i in axes(conn, 1)
    R[conn[i]] += R_el[i]
  end
  return nothing
end

"""
$(TYPEDSIGNATURES)
Assembly method for residuals
"""
function assemble!(
  global_val::NodalField, 
  fspace, block_num, e, local_val
)
  dof_conn = dof_connectivity(fspace, e)
  FiniteElementContainers.assemble!(global_val, local_val, dof_conn)
  return nothing
end

"""
$(TYPEDSIGNATURES)
generic assembly method that directly goes into a vector
for doing a residual and matrix vector product
at once
"""
function assemble!(
  R::V1, Kv::V1, R_el::V2, Kv_el::V2, conn::V3
) where {V1 <: AbstractArray{<:Number}, 
         V2 <: AbstractArray{<:Number},
         V3 <: AbstractArray{<:Integer}}

  for i in axes(conn, 1)
    R[conn[i]] += R_el[i]
    Kv[conn[i]] += Kv_el[i]
  end
  return nothing
end

# """
# $(TYPEDSIGNATURES)
# assembly method for just a residual vector

# TODO need to add an Atomix lock here
# TODO add block_id to fspace or something like that
# """
# function assemble_atomic!(
#   R::V1,
#   R_el::V2, conn::V3
# ) where {V1 <: NodalField, V2 <: AbstractVector{<:Number}, V3 <: AbstractVector{<:Integer}}

#   for i in axes(conn, 1)
#     Atomix.@atomic R.vals[conn[i]] += R_el[i]
#   end
#   return nothing
# end

# """
# $(TYPEDSIGNATURES)
# method that assumes first dof
# TODO move sorting of nodes up stream
# TODO remove other scratch unknowns and unknown_dofs arrays
# """
# function update_unknown_dofs!(
#   assembler::StaticAssembler,
#   dof,
#   fspaces, 
#   nodes_in::V
# ) where V <: AbstractVector{<:Integer}

#   # make this an assumption of the method
#   nodes = sort(nodes_in)

#   n_total_dofs = num_dofs_per_node(dof) * num_nodes(dof) - length(nodes)

#   # TODO change to a good sizehint!
#   resize!(assembler.Is, 0)
#   resize!(assembler.Js, 0)
#   resize!(assembler.unknown_dofs, 0)

#   n = 1
#   for fspace in fspaces
#     for e in 1:num_elements(fspace)
#       conn = dof_connectivity(fspace, e)
#       for temp in Iterators.product(conn, conn)
#         if insorted(temp[1], nodes) || insorted(temp[2], nodes)
#           # really do nothing here
#         else
#           push!(assembler.Is, temp[1] - count(x -> x < temp[1], nodes))
#           push!(assembler.Js, temp[2] - count(x -> x < temp[2], nodes))
#           push!(assembler.unknown_dofs, n)
#         end
#         n += 1
#       end
#     end
#   end

#   # resize cache arrays
#   resize!(assembler.klasttouch, n_total_dofs)
#   resize!(assembler.csrrowptr, n_total_dofs + 1)
#   resize!(assembler.csrcolval, length(assembler.Is))
#   resize!(assembler.csrnzval, length(assembler.Is))

#   resize!(assembler.stiffnesses, length(assembler.Is))

#   return nothing
# end

# implementations
include("DynamicAssemblers.jl")
include("MatrixFreeStaticAssembler.jl")
include("StaticAssembler.jl")

"""
$(TYPEDSIGNATURES)
method that assumes first dof
TODO move sorting of nodes up stream
TODO remove other scratch unknowns and unknown_dofs arrays
"""
function update_unknown_dofs!(
  assembler::Union{DynamicAssembler, StaticAssembler},
  dof,
  fspaces, 
  nodes_in::V
) where V <: AbstractVector{<:Integer}

  # make this an assumption of the method
  nodes = sort(nodes_in)

  n_total_dofs = num_dofs_per_node(dof) * num_nodes(dof) - length(nodes)

  # TODO change to a good sizehint!
  resize!(assembler.Is, 0)
  resize!(assembler.Js, 0)
  resize!(assembler.unknown_dofs, 0)

  n = 1
  for fspace in fspaces
    for e in 1:num_elements(fspace)
      conn = dof_connectivity(fspace, e)
      for temp in Iterators.product(conn, conn)
        if insorted(temp[1], nodes) || insorted(temp[2], nodes)
          # really do nothing here
        else
          push!(assembler.Is, temp[1] - count(x -> x < temp[1], nodes))
          push!(assembler.Js, temp[2] - count(x -> x < temp[2], nodes))
          push!(assembler.unknown_dofs, n)
        end
        n += 1
      end
    end
  end

  # resize cache arrays
  resize!(assembler.klasttouch, n_total_dofs)
  resize!(assembler.csrrowptr, n_total_dofs + 1)
  resize!(assembler.csrcolval, length(assembler.Is))
  resize!(assembler.csrnzval, length(assembler.Is))

  # resize!(assembler.stiffnesses, length(assembler.Is))

  return nothing
end