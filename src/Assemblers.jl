abstract type Assembler{Rtype, Itype} end
int_type(::Assembler{R, I}) where {R, I} = I
float_type(::Assembler{R, I}) where {R, I} = R

struct StaticAssembler{
  Rtype, Itype,
  I       <: AbstractArray{Itype, 1},
  J       <: AbstractArray{Itype, 1},
  U       <: AbstractArray{Itype, 1},
  Sizes   <: AbstractArray{Itype, 1},
  Offsets <: AbstractArray{Itype, 1},
  R       <: AbstractArray{Rtype}, # can maybe be a nodalfield depending upon what the user wants
  K       <: AbstractArray{Rtype, 1} # should always be a vector type thing
} <: Assembler{Rtype, Itype}
  Is::I
  Js::J
  unknown_indices::U
  block_sizes::Sizes
  block_offsets::Offsets
  residuals::R
  stiffnesses::K
end

"""
Default initialization
Assumes no dirichlet bcs

TODO add typing to constructor
"""
function StaticAssembler(dof::DofManager, fspaces::Fs) where Fs

  # first get total number of entries in a stupid matter
  n_entries = 0
  block_sizes = Vector{Int64}(undef, length(fspaces))
  block_offsets = Vector{Int64}(undef, length(fspaces))
  for (n, fspace) in enumerate(fspaces)
    conn = dof_connectivity(fspace)
    n_entries += size(conn, 1)^2 * size(conn, 2)
    block_sizes[n] = size(conn, 2)
    block_offsets[n] = size(conn, 1)^2
  end

  # setup pre-allocated arrays based on number of entries found above
  Is = Vector{Int64}(undef, n_entries)
  Js = Vector{Int64}(undef, n_entries)
  unknown_indices = Vector{Int64}(undef, n_entries)

  # now loop over function spaces and elements
  n = 1
  for fspace in fspaces
    for e in 1:num_elements(fspace)
      conn = dof_connectivity(fspace, e)
      for temp in Iterators.product(conn, conn)
        Is[n] = temp[1]
        Js[n] = temp[2]
        unknown_indices[n] = n
        n += 1
      end
    end
  end

  residuals = create_fields(dof)
  stiffnesses = zeros(Float64, size(Is))

  return StaticAssembler{
    Float64, Int64, 
    typeof(Is), typeof(Js), typeof(unknown_indices), typeof(block_sizes), typeof(block_offsets),
    typeof(residuals), typeof(stiffnesses)
  }(Is, Js, unknown_indices, block_sizes, block_offsets, residuals, stiffnesses)
end

"""
"""
function SparseArrays.sparse(assembler::StaticAssembler)
  ids = assembler.unknown_indices
  # return sparse(assembler.Is[ids], assembler.Js[ids], assembler.stiffnesses[ids])
  return sparse(assembler.Is, assembler.Js, assembler.stiffnesses[ids])
end

"""
assembly method for just a residual vector

TODO need to add an Atomix lock here
TODO add block_id to fspace or something like that
"""
function assemble!(
  assembler::StaticAssembler,
  R_el::V, conn
) where V <: AbstractVector

  for i in axes(conn, 1)
    assembler.residuals[conn[i]] += R_el[i]
  end
end


"""
assembly for stiffness matrix
"""
function assemble!(
  assembler::StaticAssembler,
  K_el::M, block_id::Int, el_id::Int
) where M <: AbstractMatrix

  # first get mapping from block and element id to ids in assembler.stiffnesses
  start_id = (block_id - 1) * assembler.block_sizes[block_id] + 
             (el_id - 1) * assembler.block_offsets[block_id] + 1
  end_id = start_id + assembler.block_offsets[block_id] - 1
  ids = start_id:end_id

  # now assemble into stiffnesses
  assembler.stiffnesses[ids] = K_el
end

"""
Simple method for assembling in serial
"""
function assemble!(
  assembler::StaticAssembler,
  dof::DofManager,
  fspace::FunctionSpace,
  X, U, block_id,
  residual_func, tangent_func
)

  NDof = num_dofs_per_node(dof)
  N    = num_nodes_per_element(fspace)
  NxNDof = N * NDof

  for e in 1:num_elements(fspace)
    U_el = element_level_fields(fspace, U, e)
    R_el = zeros(SVector{NxNDof, Float64})
    K_el = zeros(SMatrix{NxNDof, NxNDof, Float64, NxNDof * NxNDof})

    # quadrature loop
    for q in 1:num_q_points(fspace)
      fspace_values = getindex(fspace, X, q, e)
      R_el = R_el + residual_func(fspace_values, U_el)
      K_el = K_el + tangent_func(fspace_values, U_el)
    end

    # assemble residual using connectivity here
    conn = dof_connectivity(fspace, e)
    assemble!(assembler, R_el, conn)

    # assemble stiffness
    assemble!(assembler, K_el, block_id, e)
  end

end

function assemble!(
  assembler::StaticAssembler,
  dof::DofManager,
  fspaces, X, U,
  residual_func, tangent_func
)

  # reset in some way
  assembler.residuals .= 0.
  assembler.stiffnesses .= 0.

  for (block_id, fspace) in enumerate(fspaces)
    assemble!(assembler, dof, fspace, X, U, block_id, residual_func, tangent_func)
  end

end

"""
method that assumes first dof
TODO move sorting of nodes up stream
TODO remove other scratch unknowns and unknown_indices arrays
"""
function update_unknown_ids!(
  assembler::StaticAssembler,
  fspaces, 
  nodes_in
)

  # make this an assumption of the method
  nodes = sort(nodes_in)

  # TODO change to a good sizehint!
  resize!(assembler.Is, 0)
  resize!(assembler.Js, 0)
  resize!(assembler.unknown_indices, 0)

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
          push!(assembler.unknown_indices, n)
        end
        n += 1
      end
    end
  end
end