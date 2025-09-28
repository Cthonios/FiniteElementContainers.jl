"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
General sparse matrix assembler that can handle first or second order
problems in time. 
"""
struct SparseMatrixAssembler{
  Dof      <: DofManager, 
  Pattern  <: SparsityPattern, 
  Storage1 <: AbstractArray{<:Number, 1},
  Storage2 <: AbstractField,
  Storage3 <: NamedTuple
} <: AbstractAssembler{Dof}
  dof::Dof
  pattern::Pattern
  constraint_storage::Storage1
  damping_storage::Storage1
  hessian_storage::Storage1
  mass_storage::Storage1
  residual_storage::Storage2
  residual_unknowns::Storage1
  scalar_quadrature_storage::Storage3 # useful for energy like calculations
  stiffness_storage::Storage1
  stiffness_action_storage::Storage2
  stiffness_action_unknowns::Storage1
end

# TODO this will not work for other than single H1 spaces
"""
$(TYPEDSIGNATURES)
Construct a ```SparseMatrixAssembler``` for a specific field type, 
e.g. ```H1Field```.
Can be used to create block arrays for mixed FEM problems.
"""
function SparseMatrixAssembler(dof::DofManager)
  pattern = SparsityPattern(dof)
  # constraint_storage = zeros(length(dof))
  # ND, NN = num_dofs_per_node(dof), num_nodes(dof)
  ND, NN = size(dof)
  n_total_dofs = ND * NN
  constraint_storage = zeros(n_total_dofs)
  # constraint_storage = zeros(_dof_manager_vars(dof, type))
  constraint_storage[dof.dirichlet_dofs] .= 1.
  # fill!(constraint_storage, )
  # residual_storage = zeros(length(dof))
  damping_storage = zeros(num_entries(pattern))
  hessian_storage = zeros(num_entries(pattern))
  mass_storage = zeros(num_entries(pattern))
  residual_storage = create_field(dof)
  residual_unknowns = create_unknowns(dof)
  stiffness_storage = zeros(num_entries(pattern))
  stiffness_action_storage = create_field(dof)
  stiffness_action_unknowns = create_unknowns(dof)

  # setup quadrature scalar storage
  # fspace = values(dof.H1_vars)[1].fspace
  fspace = function_space(dof)
  scalar_quadarature_storage = Matrix{Float64}[]
  for (key, val) in pairs(fspace.ref_fes)
    NQ = ReferenceFiniteElements.num_quadrature_points(val)
    NE = size(getfield(fspace.elem_conns, key), 2)
    syms = map(x -> Symbol("quadrature_field_$x"), 1:NQ)
    field = L2ElementField(zeros(Float64, NQ, NE))
    push!(scalar_quadarature_storage, field)
    # push!(scalar_quadarature_storage, zeros(Float64, NQ, NE))
  end
  scalar_quadarature_storage = NamedTuple{keys(fspace.ref_fes)}(tuple(scalar_quadarature_storage...))

  return SparseMatrixAssembler(
    dof, pattern, 
    constraint_storage, 
    damping_storage, 
    hessian_storage,
    mass_storage,
    residual_storage, residual_unknowns,
    scalar_quadarature_storage,
    stiffness_storage, 
    stiffness_action_storage, stiffness_action_unknowns
  )
end

function SparseMatrixAssembler(var::AbstractFunction; use_condensed::Bool = false)
  dof = DofManager(var; use_condensed=use_condensed)
  return SparseMatrixAssembler(dof)
end

function Base.show(io::IO, asm::SparseMatrixAssembler)
  println(io, "SparseMatrixAssembler")
  println(io, "  ", asm.dof)
end

"""
$(TYPEDSIGNATURES)
Specialization of of ```_assemble_element!``` for ```SparseMatrixAssembler```.
"""
function _assemble_element!(
  pattern::SparsityPattern, storage, K_el::SMatrix, el_id::Int, block_id::Int
)
  # figure out ids needed to update
  block_start_index = values(pattern.block_start_indices)[block_id]
  block_el_level_size = values(pattern.block_el_level_sizes)[block_id]
  start_id = block_start_index + 
             (el_id - 1) * block_el_level_size
  end_id = start_id + block_el_level_size - 1
  ids = start_id:end_id

  # get appropriate storage and update values
  @views storage[ids] += K_el[:]
  return nothing
end

# TODO this only work on CPU right now
function _adjust_matrix_entries_for_constraints!(
  A::SparseMatrixCSC, constraint_storage, ::KA.CPU;
  penalty_scale = 1.e6
)
  # first ensure things are the right size
  @assert size(A, 1) == size(A, 2)
  @assert length(constraint_storage) == size(A, 2)

  # hacky for now
  # need a penalty otherwise we get into trouble with
  # iterative linear solvers even for a simple poisson problem
  # TODO perhaps this should be optional somehow
  penalty = penalty_scale * tr(A) / size(A, 2)

  # now modify A => (I - G) * A + G
  nz = nonzeros(A)
  rowval = rowvals(A)
  for j in 1:size(A, 2)
    col_start = A.colptr[j]
    col_end   = A.colptr[j + 1] - 1
    for k in col_start:col_end
      # for (I - G) * A term
      nz[k] = (1. - constraint_storage[j]) * nz[k]

      # for + G term
      if rowval[k] == j
        @inbounds nz[k] = nz[k] + penalty * constraint_storage[j]
      end
    end
  end

  return nothing
end

# COV_EXCL_START
KA.@kernel function _adjust_matrix_entries_for_constraints_kernel!(
  A, constraint_storage, trA;
  penalty_scale = 1.e6
)
  J = KA.@index(Global)

  penalty = penalty_scale * trA / size(A, 2)

  # now modify A => (I - G) * A + G
  nz = nonzeros(A)
  rowval = rowvals(A)

  col_start = A.colptr[J]
  col_end   = A.colptr[J + 1] - 1
  for k in col_start:col_end
    # for (I - G) * A term
    nz[k] = (1. - constraint_storage[J]) * nz[k]

    # for + G term
    if rowval[k] == J
      @inbounds nz[k] = nz[k] + penalty * constraint_storage[J]
    end
  end
end
# COV_EXCL_STOP

function _adjust_matrix_entries_for_constraints(
  A, constraint_storage, backend::KA.Backend
)
  # first ensure things are the right size
  @assert size(A, 1) == size(A, 2)
  @assert length(constraint_storage) == size(A, 2)

  # get trA ahead of time to save some allocations at kernel level
  trA = tr(A)
  
  kernel! = _adjust_matrix_entries_for_constraints_kernel!(backend)
  kernel!(A, constraint_storage, trA, ndrange = size(A, 2))
  return nothing
end

function _hessian(assembler::SparseMatrixAssembler, ::KA.CPU)
  H = SparseArrays.sparse!(assembler.pattern, assembler.hessian_storage)

  if _is_condensed(assembler.dof)
    _adjust_matrix_entries_for_constraints!(H, assembler.constraint_storage, KA.get_backend(assembler))
  end

  return H
end

function _mass(assembler::SparseMatrixAssembler, ::KA.CPU)
  M = SparseArrays.sparse!(assembler.pattern, assembler.mass_storage)

  if _is_condensed(assembler.dof)
    _adjust_matrix_entries_for_constraints!(M, assembler.constraint_storage, KA.get_backend(assembler))
  end

  return M
end

function _stiffness(assembler::SparseMatrixAssembler, ::KA.CPU)
  K = SparseArrays.sparse!(assembler.pattern, assembler.stiffness_storage)

  if _is_condensed(assembler.dof)
    _adjust_matrix_entries_for_constraints!(K, assembler.constraint_storage, KA.get_backend(assembler))
  end

  return K
end

# TODO probably only works for H1 fields
# TODO Need to specialize below for different field types
# TODO make keyword use_condensed more clear
# the use case here being to flag how to update the sparsity pattern
# constraint_storage is used to make a diagonal matrix of 1s and 0s to zero out element of
# the residual and stiffness appropriately without having to reshape, Is, Js, etc.
# when we want to change BCs which is slow

function update_dofs!(assembler::SparseMatrixAssembler, dirichlet_bcs)
  use_condensed = _is_condensed(assembler.dof)

  if length(dirichlet_bcs) > 0
    dirichlet_dofs = mapreduce(x -> x.dofs, vcat, dirichlet_bcs)
    dirichlet_dofs = unique(sort(dirichlet_dofs))
  else
    dirichlet_dofs = Vector{Int}(undef, 0)
  end

  update_dofs!(assembler.dof, dirichlet_dofs)

  if use_condensed
    _update_dofs_condensed!(assembler)
  else
    _update_dofs!(assembler, dirichlet_dofs)
  end
  return nothing
end
# TODO Need to specialize below for different field types
# TODO make keyword use_condensed more clear
# the use case here being to flag how to update the sparsity pattern
# constraint_storage is used to make a diagonal matrix of 1s and 0s to zero out element of
# the residual and stiffness appropriately without having to reshape, Is, Js, etc.
# when we want to change BCs which is slow

function _update_dofs_condensed!(assembler::SparseMatrixAssembler)
  assembler.constraint_storage[assembler.dof.unknown_dofs] .= 0.
  assembler.constraint_storage[assembler.dof.dirichlet_dofs] .= 1.
  return nothing
end

# TODO part of this method should be moved to SparsityPattern.jl
# TODO specialize on field type
# TODO probably only works on H1 write now
function _update_dofs!(assembler::SparseMatrixAssembler, dirichlet_dofs::T) where T <: AbstractArray{<:Integer, 1}

  # resize the resiual unkowns
  resize!(assembler.residual_unknowns, length(assembler.dof.unknown_dofs))
  resize!(assembler.stiffness_action_unknowns, length(assembler.dof.unknown_dofs))

  _update_dofs!(assembler.pattern, assembler.dof, dirichlet_dofs)

  return nothing
end
