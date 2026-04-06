"""
$(TYPEDEF)
$(TYPEDSIGNATURES)
General sparse matrix assembler that can handle first or second order
problems in time. 
"""
struct SparseMatrixAssembler{
  Condensed,
  # NumArrDims,
  UseSparseVec,
  UseStaticArrays,
  IV           <: AbstractArray{Int, 1},
  RV           <: AbstractArray{Float64, 1},
  Var          <: AbstractFunction,
  # FieldStorage <: AbstractField{Float64, NumArrDims, RV}
  FieldStorage
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
  scalar_quadrature_storage::L2Field{Float64, RV}
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
function SparseMatrixAssembler(
  dof::DofManager;
  use_sparse_vector::Bool = false,
  use_static_arrays::Bool = true
)
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
  # residual_storage = create_assembler_cache(matrix_pattern, AssembledVector())
  residual_unknowns = create_unknowns(dof)
  stiffness_storage = zeros(num_entries(matrix_pattern))
  stiffness_action_storage = create_field(dof)
  stiffness_action_unknowns = create_unknowns(dof)

  # setup quadrature scalar storage
  fspace = function_space(dof)
  scalar_quadrature_storage = L2Field(undef, Float64, 1, block_quadrature_sizes(fspace))
  fill!(scalar_quadrature_storage, 0.0)

  return SparseMatrixAssembler{
    _is_condensed(dof),
    # num_fields(residual_storage),
    use_sparse_vector,
    use_static_arrays,
    typeof(dof.unknown_dofs),
    typeof(residual_storage.data),
    typeof(dof.var),
    typeof(residual_storage)
  }(
    dof, matrix_pattern, vector_pattern,
    constraint_storage, 
    damping_storage, 
    hessian_storage,
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

function Adapt.adapt_structure(to, asm::SparseMatrixAssembler)
  dof = adapt(to, asm.dof)
  residual_storage = adapt(to, asm.residual_storage)
  return SparseMatrixAssembler{
    _is_condensed(dof),
    # num_fields(residual_storage),
    _use_sparse_vector(asm),
    _use_static_arrays(asm),
    typeof(dof.unknown_dofs),
    typeof(residual_storage.data),
    typeof(dof.var),
    typeof(residual_storage)
  }(
    dof,
    adapt(to, asm.matrix_pattern),
    adapt(to, asm.vector_pattern),
    adapt(to, asm.constraint_storage),
    adapt(to, asm.damping_storage),
    adapt(to, asm.hessian_storage),
    adapt(to, asm.mass_storage),
    residual_storage,
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
  backend = KA.get_backend(asm)
  return KA.zeros(backend, Float64, asm.matrix_pattern.max_entries[1])
end

# TODO probably only works for H1 fields
# TODO Need to specialize below for different field types
# TODO make keyword use_condensed more clear
# the use case here being to flag how to update the sparsity pattern
# constraint_storage is used to make a diagonal matrix of 1s and 0s to zero out element of
# the residual and stiffness appropriately without having to reshape, Is, Js, etc.
# when we want to change BCs which is slow

function _use_sparse_vector(
  ::SparseMatrixAssembler{T1, UseSVec, T3, T4, T5, T6, T7}
) where {T1, UseSVec, T3, T4, T5, T6, T7}
  return UseSVec
end

function _use_static_arrays(
  ::SparseMatrixAssembler{T1, T2, UseSArrs, T4, T5, T6, T7}
) where {T1, T2, UseSArrs, T4, T5, T6, T7}
  return UseSArrs
end

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
  
    _update_dofs!(assembler.matrix_pattern, assembler.dof, ddofs)
    _update_dofs!(assembler.vector_pattern, assembler.dof, ddofs)
  end
  return nothing
end
