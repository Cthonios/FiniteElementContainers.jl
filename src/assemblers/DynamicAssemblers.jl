"""
Assembler for dynamic problems without damping

Provides both a mass and stiffness matrix
"""
struct DynamicAssembler{
  Rtype, Itype,
  I       <: AbstractArray{Itype, 1},
  J       <: AbstractArray{Itype, 1},
  U       <: AbstractArray{Itype, 1},
  Sizes   <: AbstractArray{Itype, 1},
  Offsets <: AbstractArray{Itype, 1},
  R       <: AbstractArray{Rtype}, # can maybe be a nodalfield depending upon what the user wants
  K       <: AbstractArray{Rtype, 1}, # should always be a vector type thing
  M       <: AbstractArray{Rtype, 1}
} <: Assembler{Rtype, Itype}
  Is::I
  Js::J
  unknown_dofs::U
  block_sizes::Sizes
  block_offsets::Offsets
  residuals::R
  stiffnesses::K
  masses::M
end

"""
Default initialization
Assumes no dirichlet bcs

TODO add typing to constructor
"""
function DynamicAssembler(dof::DofManager, fspaces::Fs) where Fs

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
  unknown_dofs = Vector{Int64}(undef, n_entries)

  # now loop over function spaces and elements
  n = 1
  for fspace in fspaces
    for e in 1:num_elements(fspace)
      conn = dof_connectivity(fspace, e)
      for temp in Iterators.product(conn, conn)
        Is[n] = temp[1]
        Js[n] = temp[2]
        unknown_dofs[n] = n
        n += 1
      end
    end
  end

  residuals = create_fields(dof)
  stiffnesses = zeros(Float64, size(Is))
  masses = zeros(Float64, size(Is))

  return DynamicAssembler{
    Float64, Int64, 
    typeof(Is), typeof(Js), typeof(unknown_dofs), typeof(block_sizes), typeof(block_offsets),
    typeof(residuals), typeof(stiffnesses), typeof(masses)
  }(Is, Js, unknown_dofs, block_sizes, block_offsets, residuals, stiffnesses, masses)
end

"""
"""
function SparseArrays.sparse(assembler::DynamicAssembler)
  ids = assembler.unknown_dofs
  K = sparse(assembler.Is, assembler.Js, assembler.stiffnesses[ids])
  M = sparse(assembler.Is, assembler.Js, assembler.masses[ids])
  return K, M
end

"""
assembly for stiffness matrix
"""
function assemble!(
  assembler::DynamicAssembler,
  K_el::M1, M_el::M2, block_id::Int, el_id::Int
) where {M1 <: AbstractMatrix{<:Number}, M2 <: AbstractMatrix{<:Number}}

  # first get mapping from block and element id to ids in assembler.stiffnesses
  start_id = (block_id - 1) * assembler.block_sizes[block_id] + 
             (el_id - 1) * assembler.block_offsets[block_id] + 1
  end_id = start_id + assembler.block_offsets[block_id] - 1
  ids = start_id:end_id

  # now assemble into stiffnesses
  assembler.stiffnesses[ids] = K_el
  assembler.masses[ids] = M_el
end

"""
Simple method for assembling in serial
"""
function assemble!(
  assembler::DynamicAssembler,
  dof::DofManager,
  fspace::FunctionSpace,
  X, U, block_id,
  residual_func, tangent_func, mass_func
)

  NDof = num_dofs_per_node(dof)
  N    = num_nodes_per_element(fspace)
  NxNDof = N * NDof

  for e in 1:num_elements(fspace)
    U_el = element_level_fields(fspace, U, e)
    R_el = zeros(SVector{NxNDof, Float64})
    K_el = zeros(SMatrix{NxNDof, NxNDof, Float64, NxNDof * NxNDof})
    M_el = zeros(SMatrix{NxNDof, NxNDof, Float64, NxNDof * NxNDof})

    # quadrature loop
    for q in 1:num_q_points(fspace)
      fspace_values = getindex(fspace, X, q, e)
      R_el = R_el + residual_func(fspace_values, U_el)
      K_el = K_el + tangent_func(fspace_values, U_el)
      M_el = M_el + mass_func(fspace_values, U_el)
    end

    # assemble residual using connectivity here
    conn = dof_connectivity(fspace, e)
    assemble!(assembler, R_el, conn)

    # assemble stiffness and mass
    assemble!(assembler, K_el, M_el, block_id, e)
  end

end

function assemble!(
  assembler::DynamicAssembler,
  dof::DofManager,
  fspaces, X, U,
  residual_func, tangent_func, mass_func
)

  # reset in some way
  assembler.residuals .= 0.
  assembler.stiffnesses .= 0.

  for (block_id, fspace) in enumerate(fspaces)
    assemble!(assembler, dof, fspace, X, U, block_id, residual_func, tangent_func, mass_func)
  end

end


