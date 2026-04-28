"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
General sparse matrix assembler that can handle first or second order
problems in time. 
"""
struct SparseMatrixAssembler{
  Condensed,
  SparseMatrixType,
  UseInPlaceMethods,
  UseSparseVec,
  IV           <: AbstractArray{Int, 1},
  RV           <: AbstractArray{Float64, 1},
  Var          <: AbstractFunction,
  FieldStorage
} <: AbstractAssembler{DofManager{Condensed, Int, IV, Var}}
  dof::DofManager{Condensed, Int, IV, Var}
  matrix_pattern::SparseMatrixPattern{IV, RV}
  vector_pattern::SparseVectorPattern{IV}
  constraint_storage::RV
  mass_storage::RV
  residual_storage::FieldStorage
  residual_unknowns::RV
  scalar_quadrature_storage::L2Field{Float64, RV}
  stiffness_storage::RV
  stiffness_action_storage::FieldStorage
  stiffness_action_unknowns::RV

  function SparseMatrixAssembler{
    Condensed,
    SparseMatrixType,
    UseInPlaceMethods,
    UseSparseVec
  }(
    dof, matrix_pattern, vector_pattern, constraint_storage,
    mass_storage,
    residual_storage, residual_unknowns,
    scalar_quadrature_storage,
    stiffness_storage, stiffness_action_storage, stiffness_action_unknowns
  ) where {Condensed, SparseMatrixType, UseInPlaceMethods, UseSparseVec}
    new{
      Condensed, SparseMatrixType, UseInPlaceMethods, UseSparseVec,
      typeof(dof.unknown_dofs), typeof(residual_storage.data),
      typeof(dof.var), typeof(stiffness_action_storage)
    }(
      dof, matrix_pattern, vector_pattern,
      constraint_storage,
      mass_storage,
      residual_storage, residual_unknowns,
      scalar_quadrature_storage,
      stiffness_storage,
      stiffness_action_storage, stiffness_action_unknowns
    )
  end
end

# TODO this will not work for other than single H1 spaces
"""
$(TYPEDSIGNATURES)
Construct a ```SparseMatrixAssembler``` for a specific field type, 
e.g. ```H1Field```.
Can be used to create block arrays for mixed FEM problems.
"""
function SparseMatrixAssembler(
  dof::DofManager;
  sparse_matrix_type::Symbol = :csc,
  use_inplace_methods::Bool = false,
  use_sparse_vector::Bool = false,
  matrix_free::Bool = false
)
  return SparseMatrixAssembler{
    _sym_to_sparse_matrix_type(sparse_matrix_type),
    use_inplace_methods,
    use_sparse_vector
  }(dof; matrix_free = matrix_free)
end

function SparseMatrixAssembler{SparseMatrixType, UseInPlaceMethods, UseSparseVec}(
  dof::DofManager;
  matrix_free::Bool = false
) where {SparseMatrixType, UseInPlaceMethods, UseSparseVec}
  # When matrix_free=true, the matrix-side storage (the sparse pattern and the
  # matrix-value buffers) are constructed empty.  This avoids ~7 GB of
  # allocations on a ~530 k-DOF mesh for integrators that never assemble a
  # global matrix (e.g. central difference, L-BFGS).  Calling assemble_matrix!
  # on such an assembler errors with a clear message; the mass/stiffness
  # accessors return a zero sparse matrix of the correct shape.
  matrix_pattern = matrix_free ? _empty_matrix_pattern(dof) : SparseMatrixPattern(dof)
  vector_pattern = SparseVectorPattern(dof)

  ND, NN = size(dof)
  n_total_dofs = ND * NN
  constraint_storage = zeros(n_total_dofs)
  constraint_storage[dof.dirichlet_dofs] .= 1.

  n_matrix_entries = matrix_free ? 0 : num_entries(matrix_pattern)
  mass_storage = zeros(n_matrix_entries)
  residual_storage = create_field(dof)
  residual_unknowns = create_unknowns(dof)
  stiffness_storage = zeros(n_matrix_entries)
  stiffness_action_storage = create_field(dof)
  stiffness_action_unknowns = create_unknowns(dof)

  # setup quadrature scalar storage
  fspace = function_space(dof)
  scalar_quadrature_storage = L2Field(undef, Float64, 1, block_quadrature_sizes(fspace))
  fill!(scalar_quadrature_storage, 0.0)

  return SparseMatrixAssembler{
    _is_condensed(dof),
    SparseMatrixType,
    UseInPlaceMethods,
    UseSparseVec
  }(
    dof, matrix_pattern, vector_pattern,
    constraint_storage,
    mass_storage,
    residual_storage, residual_unknowns,
    scalar_quadrature_storage,
    stiffness_storage,
    stiffness_action_storage, stiffness_action_unknowns
  )
end

function SparseMatrixAssembler(
  var::AbstractFunction;
  use_condensed::Bool = false,
  kwargs...
)
  dof = DofManager(var; use_condensed = use_condensed)
  return SparseMatrixAssembler(dof; kwargs...)
end

# Construct a SparseMatrixPattern with all internal arrays empty.  Used as a
# placeholder for matrix-free assemblers; carries no sparsity information and
# cannot be used by SparseArrays.sparse! or block_view.
function _empty_matrix_pattern(dof::DofManager)
  IV = typeof(dof.unknown_dofs)
  return SparseMatrixPattern{IV, Vector{Float64}}(
    similar(dof.unknown_dofs, 0),  # Is
    similar(dof.unknown_dofs, 0),  # Js
    similar(dof.unknown_dofs, 0),  # unknown_dofs
    Int[],                          # block_start_indices
    [0],                            # max_entries  (single zero — see create_assembler_cache)
    similar(dof.unknown_dofs, 0),  # klasttouch
    similar(dof.unknown_dofs, 0),  # csrrowptr
    similar(dof.unknown_dofs, 0),  # csrcolval
    Float64[],                      # csrnzval
    similar(dof.unknown_dofs, 0),  # csccolptr
    similar(dof.unknown_dofs, 0),  # cscrowval
    Float64[],                      # cscnzval
    similar(dof.unknown_dofs, 0)   # permutation
  )
end

# True if this assembler was constructed with matrix_free=true: the sparsity
# pattern and matrix-value buffers are empty and assemble_matrix! is forbidden.
_is_matrix_free(asm::SparseMatrixAssembler) = isempty(asm.matrix_pattern.Is)

function Adapt.adapt_structure(to, asm::SparseMatrixAssembler)
  return SparseMatrixAssembler{
    _is_condensed(asm.dof),
    _sparse_matrix_type(asm),
    _use_inplace_methods(asm),
    _use_sparse_vector(asm),
  }(
    adapt(to, asm.dof),
    adapt(to, asm.matrix_pattern),
    adapt(to, asm.vector_pattern),
    adapt(to, asm.constraint_storage),
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

function create_assembler_cache(asm::SparseMatrixAssembler, ::AssembledSparseVector)
  backend = KA.get_backend(asm)
  return KA.zeros(backend, Float64, asm.vector_pattern.max_entries[1])
end

function create_assembler_cache(asm::SparseMatrixAssembler, ::AssembledMatrix)
  if _is_matrix_free(asm)
    error("create_assembler_cache(::AssembledMatrix) called on a matrix-free " *
          "SparseMatrixAssembler.  Re-create the assembler with matrix_free=false.")
  end
  backend = KA.get_backend(asm)
  return KA.zeros(backend, Float64, asm.matrix_pattern.max_entries[1])
end

# helper methods for accessing parametric types
function _sparse_matrix_type(
  ::SparseMatrixAssembler{T1, SparseMatrixType, T3, T4, T5, T6, T7, T8}
) where {T1, SparseMatrixType, T3, T4, T5, T6, T7, T8}
  return SparseMatrixType
end

function _use_inplace_methods(
  ::SparseMatrixAssembler{T1, T2, UseInPlaceMethods, T4, T5, T6, T7, T8}
) where {T1, T2, UseInPlaceMethods, T4, T5, T6, T7, T8}
  return UseInPlaceMethods
end

function _use_sparse_vector(
  ::SparseMatrixAssembler{T1, T2, T3, UseSparseVec, T5, T6, T7, T8}
) where {T1, T2, T3, UseSparseVec, T5, T6, T7, T8}
  return UseSparseVec
end

# TODO probably only works for H1 fields
# TODO Need to specialize below for different field types
# TODO make keyword use_condensed more clear
# the use case here being to flag how to update the sparsity pattern
# constraint_storage is used to make a diagonal matrix of 1s and 0s to zero out element of
# the residual and stiffness appropriately without having to reshape, Is, Js, etc.
# when we want to change BCs which is slow
function update_dofs!(assembler::AbstractAssembler, dirichlet_bcs::DirichletBCs)
  use_condensed = _is_condensed(assembler.dof)

  if length(dirichlet_bcs) > 0
    ddofs = dirichlet_dofs(dirichlet_bcs)
  else
    ddofs = Vector{Int}(undef, 0)
  end

  update_dofs!(assembler.dof, ddofs)

  # TODO make keyword use_condensed more clear
  # the use case here being to flag how to update the sparsity pattern
  # constraint_storage is used to make a diagonal matrix of 1s and 0s to zero out element of
  # the residual and stiffness appropriately without having to reshape, Is, Js, etc.
  # # when we want to change BCs which is slow
  if use_condensed
    assembler.constraint_storage[assembler.dof.unknown_dofs] .= 0.
    assembler.constraint_storage[assembler.dof.dirichlet_dofs] .= 1.
  else
    resize!(assembler.residual_unknowns, length(assembler.dof.unknown_dofs))
    resize!(assembler.stiffness_action_unknowns, length(assembler.dof.unknown_dofs))

    # Skip matrix-pattern update on matrix-free assemblers — _update_dofs!
    # would otherwise rebuild the pattern from scratch and silently flip the
    # assembler back into the matrix-bearing mode.
    if assembler isa SparseMatrixAssembler && _is_matrix_free(assembler)
      # no-op: matrix pattern stays empty
    else
      _update_dofs!(assembler.matrix_pattern, assembler.dof, ddofs)
    end
    _update_dofs!(assembler.vector_pattern, assembler.dof, ddofs)
  end
  return nothing
end
