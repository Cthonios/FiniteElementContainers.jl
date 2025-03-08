"""
$(TYPEDEF)
$(TYPEDFIELDS)
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
  R       <: NodalField,
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

    # hacky for now
    if eltype(conn) <: SVector
      n_entries += length(conn[1])^2 * length(conn)
      block_sizes[n] = length(conn)
      block_offsets[n] = length(conn[1])^2
    else
      n_entries += size(conn, 1)^2 * size(conn, 2)
      block_sizes[n] = size(conn, 2)
      block_offsets[n] = size(conn, 1)^2
    end
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

  residuals = create_fields(dof) #|> vec
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
    Is, Js, unknown_dofs, block_sizes, block_offsets, 
    # fields
    residuals, stiffnesses,
    # cache arrays
    klasttouch, csrrowptr, csrcolval, csrnzval,
    # additional cache arrays
    csccolptr, cscrowval, cscnzval
  )
end

"""
$(TYPEDSIGNATURES)
"""
function SparseArrays.sparse(assembler::StaticAssembler)
  ids = assembler.unknown_dofs
  return @views sparse(assembler.Is, assembler.Js, assembler.stiffnesses[ids])
end

"""
$(TYPEDSIGNATURES)
"""
function SparseArrays.sparse!(assembler::StaticAssembler)
  ids = assembler.unknown_dofs
  return @views SparseArrays.sparse!(
    assembler.Is, assembler.Js, assembler.stiffnesses[ids],
    length(assembler.klasttouch), length(assembler.klasttouch), +, assembler.klasttouch,
    assembler.csrrowptr, assembler.csrcolval, assembler.csrnzval,
    assembler.csccolptr, assembler.cscrowval, assembler.cscnzval
  )
end

"""
$(TYPEDSIGNATURES)
"""
function Base.similar(asm::StaticAssembler)
  residuals = similar(asm.residuals)
  stiffnesses = similar(asm.stiffnesses)
  # TODO should probably remove below lines
  residuals .= zero(eltype(residuals))
  stiffnesses .= zero(eltype(stiffnesses))
  StaticAssembler{
    eltype(residuals), Int64,
    typeof(asm.Is), typeof(asm.Js), typeof(asm.unknown_dofs), 
    typeof(asm.block_sizes), typeof(asm.block_offsets),
    typeof(asm.residuals), typeof(asm.stiffnesses),
    typeof(asm.klasttouch), typeof(asm.csrrowptr), 
    typeof(asm.csrcolval), typeof(asm.csrnzval),
    typeof(asm.csccolptr), typeof(asm.cscrowval), typeof(asm.cscnzval)
  }(
    copy(asm.Is), copy(asm.Js), copy(asm.unknown_dofs),
    copy(asm.block_sizes), copy(asm.block_offsets),
    residuals, stiffnesses,
    copy(asm.klasttouch), copy(asm.csrrowptr),
    copy(asm.csrcolval), copy(asm.csrnzval),
    copy(asm.csccolptr), copy(asm.cscrowval), copy(asm.cscnzval)
  )
end

"""
$(TYPEDSIGNATURES)
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
  # old behavior that wouldn't work for seperate quadrature points sepraturely
  # assembler.stiffnesses[ids] = K_el
  assembler.stiffnesses[ids] += K_el[:]
end

"""
$(TYPEDSIGNATURES)
"""
function assemble!(
  global_val::StaticAssembler, 
  fspace, block_num, e, local_val
)
  FiniteElementContainers.assemble!(global_val, local_val, block_num, e)
  return nothing
end

# """
# $(TYPEDSIGNATURES)
# assembly for stiffness matrix
# """
# function assemble_atomic!(
#   assembler::StaticAssembler,
#   K_el::M, block_id::Int, el_id::Int
# ) where M <: AbstractMatrix

#   # first get mapping from block and element id to ids in assembler.stiffnesses
#   start_id = (block_id - 1) * assembler.block_sizes[block_id] + 
#              (el_id - 1) * assembler.block_offsets[block_id] + 1
#   end_id = start_id + assembler.block_offsets[block_id] - 1
#   ids = start_id:end_id

#   # now assemble into stiffnesses
#   # Atomix.@atomic assembler.stiffnesses[ids] = K_el
#   for (n, id) in enumerate(ids)
#     Atomix.@atomic assembler.stiffnesses[id] = K_el[n]
#   end
# end

"""
$(TYPEDSIGNATURES)
Simple method for assembling in serial
"""
# TODO figure this one out
function assemble!(
  R,
  assembler::StaticAssembler,
  dof::DofManager,
  fspace::AbstractFunctionSpace,
  X, U, block_id,
  residual_func, tangent_func
)

  NDof = num_dofs_per_node(dof)
  N    = num_nodes_per_element(fspace)
  NxNDof = N * NDof

  R = assembler.residuals

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
    assemble!(R, R_el, conn)

    # assemble stiffness
    assemble!(assembler, K_el, block_id, e)
  end

end

"""
$(TYPEDSIGNATURES)
Top level method using methods
"""
# TODO figure this one out
function assemble!(
  # R, 
  assembler::StaticAssembler,
  dof::DofManager,
  fspaces, X, U,
  residual_func, tangent_func
)

  # reset in some way
  # assembler.residuals .= 0.
  R = assembler.residuals
  R .= zero(eltype(R))
  assembler.stiffnesses .= zero(eltype(assembler.stiffnesses))

  for (block_id, fspace) in enumerate(fspaces)
    assemble!(R, assembler, dof, fspace, X, U, block_id, residual_func, tangent_func)
  end

end

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

#   # resize!(assembler.stiffnesses, length(assembler.Is))

#   return nothing
# end
