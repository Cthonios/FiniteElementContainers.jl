"""
Assembler for static or quasistatic problems where
only a stiffness matrix is necessary
"""
struct StaticAssembler{
  Rtype, Itype,
  I       <: AbstractArray{Itype, 1},
  J       <: AbstractArray{Itype, 1},
  U       <: AbstractArray{Itype, 1},
  Sizes   <: AbstractArray{Itype, 1},
  Offsets <: AbstractArray{Itype, 1},
  R       <: AbstractArray{Rtype}, # can maybe be a nodalfield depending upon what the user wants
  K       <: AbstractArray{Rtype, 1}, # should always be a vector type thing
  # cache types
  C1, C2, C3, C4,
  # additional cache arrays
  C5, C6, C7
} <: Assembler{Rtype, Itype}
  Is::I
  Js::J
  unknown_dofs::U
  block_sizes::Sizes
  block_offsets::Offsets
  residuals::R
  stiffnesses::K
  # cache arrays
  klasttouch::C1
  csrrowptr::C2
  csrcolval::C3
  csrnzval::C4
  # additional cache arrays
  csccolptr::C5
  cscrowval::C6
  cscnzval::C7
end

"""
Default initialization
Assumes no dirichlet bcs

TODO add typing to constructor
"""
function StaticAssembler(dof::DofManager, fspaces::Fs) where Fs

  # get number of dofs for creating cache arrays
  n_total_dofs = num_dofs_per_node(dof) * num_nodes(dof)

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

  # create caches
  klasttouch = zeros(Int64, n_total_dofs)
  csrrowptr  = zeros(Int64, n_total_dofs + 1)
  csrcolval  = zeros(Int64, length(Is))
  csrnzval   = zeros(Float64, length(Is))

  csccolptr = Vector{Int64}(undef, 0)
  cscrowval = Vector{Int64}(undef, 0)
  cscnzval  = Vector{Float64}(undef, 0)

  return StaticAssembler{
    Float64, Int64, 
    typeof(Is), typeof(Js), typeof(unknown_dofs), typeof(block_sizes), typeof(block_offsets),
    typeof(residuals), typeof(stiffnesses),
    # cache arrays
    typeof(klasttouch), typeof(csrrowptr), typeof(csrcolval), typeof(csrnzval),
    # additional cache arrays
    typeof(csccolptr), typeof(cscrowval), typeof(cscnzval)
  }(
    Is, Js, unknown_dofs, block_sizes, block_offsets, residuals, stiffnesses,
    # cache arrays
    klasttouch, csrrowptr, csrcolval, csrnzval,
    # additional cache arrays
    csccolptr, cscrowval, cscnzval
  )
end

"""
"""
function SparseArrays.sparse(assembler::StaticAssembler)
  ids = assembler.unknown_dofs
  return sparse(assembler.Is, assembler.Js, assembler.stiffnesses[ids])
end

"""
"""
function SparseArrays.sparse!(assembler::StaticAssembler)
  ids = assembler.unknown_dofs
  return SparseArrays.sparse!(
    assembler.Is, assembler.Js, assembler.stiffnesses[ids],
    length(assembler.klasttouch), length(assembler.klasttouch), +, assembler.klasttouch,
    assembler.csrrowptr, assembler.csrcolval, assembler.csrnzval,
    assembler.csccolptr, assembler.cscrowval, assembler.cscnzval
  )
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


