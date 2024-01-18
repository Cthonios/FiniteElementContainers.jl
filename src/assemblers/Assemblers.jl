"""
"""
abstract type Assembler{Rtype, Itype} end

"""
"""
int_type(::Assembler{R, I}) where {R, I} = I

"""
"""
float_type(::Assembler{R, I}) where {R, I} = R


"""
assembly method for just a residual vector

TODO need to add an Atomix lock here
TODO add block_id to fspace or something like that
"""
function assemble!(
  assembler::Assembler,
  R_el::V1, conn::V2
) where {V1 <: AbstractVector{<:Number}, V2 <: AbstractVector{<:Integer}}

  for i in axes(conn, 1)
    assembler.residuals[conn[i]] += R_el[i]
  end
end

"""
method that assumes first dof
TODO move sorting of nodes up stream
TODO remove other scratch unknowns and unknown_dofs arrays
"""
function update_unknown_dofs!(
  assembler::Assembler,
  fspaces, 
  nodes_in::V
) where V <: AbstractVector{<:Integer}

  # make this an assumption of the method
  nodes = sort(nodes_in)

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
end

# implementations
include("DynamicAssemblers.jl")
include("StaticAssembler.jl")
