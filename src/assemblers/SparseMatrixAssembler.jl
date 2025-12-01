"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
General sparse matrix assembler that can handle first or second order
problems in time. 
"""
struct SparseMatrixAssembler{
  Condensed,
  NumArrDims,
  NumFields,
  IV                <: AbstractArray{Int, 1},
  RV                <: AbstractArray{Float64, 1},
  Var               <: AbstractFunction,
  FieldStorage      <: AbstractField{Float64, NumArrDims, RV, NumFields}, # should we make 2 not hardcoded? E.g. for 
  QuadratureStorage <: NamedTuple
} <: AbstractAssembler{DofManager{Condensed, Int, IV, Var}}
  dof::DofManager{Condensed, Int, IV, Var}
  matrix_pattern::SparseMatrixPattern{IV, RV}
  vector_pattern::SparseVectorPattern{IV}
  constraint_storage::RV
  damping_storage::RV
  hessian_storage::RV
  mass_storage::RV
  residual_storage::FieldStorage
  residual_unknowns::RV
  scalar_quadrature_storage::QuadratureStorage # useful for energy like calculations
  stiffness_storage::RV
  stiffness_action_storage::FieldStorage
  stiffness_action_unknowns::RV
end

# TODO this will not work for other than single H1 spaces
"""
$(TYPEDSIGNATURES)
Construct a ```SparseMatrixAssembler``` for a specific field type, 
e.g. ```H1Field```.
Can be used to create block arrays for mixed FEM problems.
"""
function SparseMatrixAssembler(dof::DofManager)
  matrix_pattern = SparseMatrixPattern(dof)
  vector_pattern = SparseVectorPattern(dof)

  ND, NN = size(dof)
  n_total_dofs = ND * NN
  constraint_storage = zeros(n_total_dofs)
  constraint_storage[dof.dirichlet_dofs] .= 1.

  damping_storage = zeros(num_entries(matrix_pattern))
  hessian_storage = zeros(num_entries(matrix_pattern))
  mass_storage = zeros(num_entries(matrix_pattern))
  residual_storage = create_field(dof)
  residual_unknowns = create_unknowns(dof)
  stiffness_storage = zeros(num_entries(matrix_pattern))
  stiffness_action_storage = create_field(dof)
  stiffness_action_unknowns = create_unknowns(dof)

  # setup quadrature scalar storage
  # fspace = values(dof.H1_vars)[1].fspace
  fspace = function_space(dof)
  scalar_quadarature_storage = Matrix{Float64}[]
  for (key, val) in pairs(fspace.ref_fes)
    NQ = ReferenceFiniteElements.num_quadrature_points(val)
    NE = size(getfield(fspace.elem_conns, key), 2)
    # NE = size(fspace.elem_conns[key], 2)
    field = L2ElementField(zeros(Float64, NQ, NE))
    push!(scalar_quadarature_storage, field)
  end
  scalar_quadarature_storage = NamedTuple{keys(fspace.ref_fes)}(tuple(scalar_quadarature_storage...))

  return SparseMatrixAssembler(
    dof, matrix_pattern, vector_pattern,
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

function Adapt.adapt_structure(to, asm::SparseMatrixAssembler)
  return SparseMatrixAssembler(
    adapt(to, asm.dof),
    adapt(to, asm.matrix_pattern),
    adapt(to, asm.vector_pattern),
    adapt(to, asm.constraint_storage),
    adapt(to, asm.damping_storage),
    adapt(to, asm.hessian_storage),
    adapt(to, asm.mass_storage),
    adapt(to, asm.residual_storage),
    adapt(to, asm.residual_unknowns),
    adapt(to, asm.scalar_quadrature_storage),
    adapt(to, asm.stiffness_storage),
    adapt(to, asm.stiffness_action_storage),
    adapt(to, asm.stiffness_action_unknowns)
  )
end

function Base.show(io::IO, asm::SparseMatrixAssembler)
  println(io, "SparseMatrixAssembler")
  println(io, "  ", asm.dof)
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
  H = SparseArrays.sparse!(assembler.matrix_pattern, assembler.hessian_storage)

  if _is_condensed(assembler.dof)
    _adjust_matrix_entries_for_constraints!(H, assembler.constraint_storage, KA.get_backend(assembler))
  end

  return H
end

function _mass(assembler::SparseMatrixAssembler, ::KA.CPU)
  M = SparseArrays.sparse!(assembler.matrix_pattern, assembler.mass_storage)

  if _is_condensed(assembler.dof)
    _adjust_matrix_entries_for_constraints!(M, assembler.constraint_storage, KA.get_backend(assembler))
  end

  return M
end

function _stiffness(assembler::SparseMatrixAssembler, ::KA.CPU)
  K = SparseArrays.sparse!(assembler.matrix_pattern, assembler.stiffness_storage)

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
    dirichlet_dofs = mapreduce(x -> x.dofs, vcat, dirichlet_bcs.bc_caches)
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

  _update_dofs!(assembler.matrix_pattern, assembler.dof, dirichlet_dofs)

  return nothing
end
