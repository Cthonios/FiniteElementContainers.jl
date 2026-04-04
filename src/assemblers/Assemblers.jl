"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
abstract type AbstractAssembler{Dof <: DofManager} end
"""
$(TYPEDSIGNATURES)
"""
KA.get_backend(asm::AbstractAssembler) = KA.get_backend(asm.dof)

abstract type AssembledReturnType end
abstract type AdditiveAssembledReturnType <: AssembledReturnType end
abstract type IndexedAssembledReturnType <: AssembledReturnType end

struct AssembledMatrix <: AdditiveAssembledReturnType
end

struct AssembledScalar <: IndexedAssembledReturnType
end

struct AssembledStruct{T} <: IndexedAssembledReturnType
end

struct AssembledVector <: AdditiveAssembledReturnType
end

struct AssembledSparseVector <: AdditiveAssembledReturnType
end

@inline function _accumulate_q_value(::AdditiveAssembledReturnType, storage, val_q, val_e, q, e)
  return val_e + val_q
end

@inline function _accumulate_q_value(::IndexedAssembledReturnType, storage, val_q, val_e, q, e)
  storage[q, e] = val_q
  return nothing
end

@inline function _accumulate_q_value(::AssembledScalar, storage::AbstractArray{T, 3}, val_q, val_e, q, e) where T
  # TODO will it always be 1 for how we're using this?
  storage[1, q, e] = val_q
  return nothing
end

# for IndexedAssembledReturnType, does nothing
function _assemble_element!(
  storage, val_q, 
  conns, # all connectivities for this element
  ::Int, ::Int
)
  return nothing
end

# assembled vector where storage is a field
function _assemble_element!(
  storage::AbstractField, R_el::SVector{NDOF, T}, 
  conns, # all connectivities for this element
  ::Int, ::Int
) where {NDOF, T <: Number}
  n_dofs = size(storage, 1)
  for d in axes(storage, 1)
    for n in axes(conns, 1)
      global_id = n_dofs * (conns[n] - 1) + d
      local_id = n_dofs * (n - 1) + d
      Atomix.@atomic storage.data[global_id] += R_el[local_id]
    end
  end
  return nothing
end

# sparse vector attempt
function _assemble_element!(
  storage::AbstractVector, R_el::SVector{NDOF, T},
  conns,
  el_id::Int,
  block_start_index::Int
) where {NDOF, T <: Number}
  # figure out ids needed to update
  start_id = block_start_index + (el_id - 1) * NDOF
  end_id = start_id + NDOF - 1
  ids = start_id:end_id
  for (i, id) in enumerate(ids)
    storage[id] += R_el.data[i]
  end
  return nothing
end

# TODO we'll need a regular matrix implementation
# as well (Can we live with 1?)
# sparse matrix
function _assemble_element!(
  storage, K_el::SMatrix{NDOF1, NDOF2, T, NDOF1xNDOF2}, 
  conns, # all connectivities for this element
  el_id::Int,
  block_start_index::Int
) where {NDOF1, NDOF2, T, NDOF1xNDOF2}
  # figure out ids needed to update
  start_id = block_start_index + (el_id - 1) * NDOF1xNDOF2
  end_id = start_id + NDOF1xNDOF2 - 1
  ids = start_id:end_id

  # get appropriate storage and update values
  for (i, id) in enumerate(ids)
    storage[id] += K_el.data[i]
  end
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
@inline function _cell_interpolants(ref_fe::R, q::Int) where R <: ReferenceFE
  return @inbounds ref_fe.cell_interps[q]
end

create_field(asm::AbstractAssembler) = create_field(asm.dof)
create_unknowns(asm::AbstractAssembler) = create_unknowns(asm.dof)

@inline function _element_level_connectivity(conns, e)
  return @views conns[:, e]
end

"""
$(TYPEDSIGNATURES)
"""
@inline function _element_level_fields(U::H1Field{T, D, NF}, ref_fe, conns, e) where {T, D, NF}
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * NF
  u_el = @views SMatrix{NNPE, NF, eltype(U), NxNDof}(U[:, conns[:, e]])
  return u_el
end

@inline function _element_level_fields(U::H1Field{T, D, NF}, ref_fe, conns) where {T, D, NF}
  NNPE = ReferenceFiniteElements.num_cell_dofs(ref_fe)
  NxNDof = NNPE * NF
  u_el = @views SMatrix{NF, NNPE, eltype(U), NxNDof}(U[:, conns])
  # u_el = @views SMatrix{NNPE, ND, eltype(U), NxNDof}(U[:, conns])
  return u_el
end

"""
$(TYPEDSIGNATURES)
"""
@inline function _element_level_fields_flat(U::H1Field{T, D, NF}, ref_fe, conns, e) where {T, D, NF}
  NNPE = ReferenceFiniteElements.num_cell_dofs(ref_fe)
  NxNDof = NNPE * NF
  u_el = @views SVector{NxNDof, eltype(U)}(U[:, conns[:, e]])
  return u_el
end

@inline function _element_level_fields_flat(U::H1Field{T, D, NF}, ref_fe, conns) where {T, D, NF}
  NNPE = ReferenceFiniteElements.num_cell_dofs(ref_fe)
  NxNDof = NNPE * NF
  u_el = @views SVector{NxNDof, eltype(U)}(U[:, conns])
  return u_el
end

@inline function element_level_fields(ref_fe, conn, X, U, U_old)
  x_el = _element_level_fields_flat(X, ref_fe, conn)
  u_el = _element_level_fields_flat(U, ref_fe, conn)
  u_el_old = _element_level_fields_flat(U_old, ref_fe, conn)
  return x_el, u_el, u_el_old
end

@inline function element_level_fields(ref_fe, conn, X, U, U_old, V)
  x_el = _element_level_fields_flat(X, ref_fe, conn)
  u_el = _element_level_fields_flat(U, ref_fe, conn)
  u_el_old = _element_level_fields_flat(U_old, ref_fe, conn)
  v_el = _element_level_fields_flat(V, ref_fe, conn)
  return x_el, u_el, u_el_old, v_el
end

"""
$(TYPEDSIGNATURES)
"""
@inline function _element_level_properties(props::AbstractArray, ::Int)
  return props
end

"""
$(TYPEDSIGNATURES)
"""
@inline function _element_level_properties(props::SVector{NP, T}, ::Int) where {NP, T}
  return props
end

"""
$(TYPEDSIGNATURES)
"""
@inline function _element_scratch(::AssembledMatrix, ref_fe, U::H1Field{T, D, NF}) where {T, D, NF}
  NNPE = ReferenceFiniteElements.num_cell_dofs(ref_fe)
  NxNDof = NNPE * NF
  K_el = zeros(SMatrix{NxNDof, NxNDof, eltype(U), NxNDof * NxNDof})
  return K_el
end

"""
$(TYPEDSIGNATURES)
"""
@inline function _element_scratch(::AssembledScalar, ref_fe, U)
  s_el = zero(eltype(U))
  return s_el
end

"""
$(TYPEDSIGNATURES)
"""
@inline function _element_scratch(::AssembledStruct, ref_fe, U)
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
@inline function _element_scratch(::AssembledVector, ref_fe, U::H1Field{T, D, NF}) where {T, D, NF}
  NNPE = ReferenceFiniteElements.num_cell_dofs(ref_fe)
  NxNDof = NNPE * NF
  R_el = zeros(SVector{NxNDof, eltype(U)})
  return R_el
end

"""
$(TYPEDSIGNATURES)
"""
function _quadrature_level_state(state::AbstractArray{<:Number, 3}, q::Int, e::Int)
  state_q = view(state, :, q, e)
  return state_q
end

"""
$(TYPEDSIGNATURES)
"""
function _surface_interpolants(ref_fe::R, q::Int, side::Int) where R <: ReferenceFE
  return @inbounds ref_fe.surf_interps[q, side]
end

# some low level sparse matrix type helpers

function _coo_matrix_constructor(backend::KA.Backend) 
  @assert false "Need to implement for $backend"
end

function _csc_matrix_constructor(backend::KA.Backend)
  @assert false "Need to implement for $backend"
end

function _coo_matrix(asm::AbstractAssembler, storage)
  backend = KA.get_backend(asm)
  return _coo_matrix(backend, asm, storage)
end

function _coo_matrix(backend::KA.Backend, asm::AbstractAssembler, storage)
  constructor = _coo_matrix_constructor(backend)

  if FiniteElementContainers._is_condensed(asm.dof)
    n_dofs = length(asm.dof)
  else
    n_dofs = length(asm.dof.unknown_dofs)
  end

  rows, cols = asm.matrix_pattern.Is, asm.matrix_pattern.Js
  perm = asm.matrix_pattern.permutation
  vals = storage[asm.matrix_pattern.unknown_dofs]
  return constructor(
    rows[perm], cols[perm], vals[perm],
    (n_dofs, n_dofs), length(asm.matrix_pattern.Is)
  )
end

function _coo_matrix(::KA.CPU, asm::AbstractAssembler, storage)
  @assert false "Currently unsupported"
end

function _csc_matrix(asm::AbstractAssembler, storage)
  backend = KA.get_backend(asm)
  return _csc_matrix(backend, asm, storage)
end

function _csc_matrix(backend::KA.Backend, asm::AbstractAssembler, storage)
  coo = _coo_matrix(backend, asm, storage)
  constructor = _csc_matrix_constructor(backend)
  return constructor(coo)
end

function _csc_matrix(::KA.CPU, asm::AbstractAssembler, storage)
  return SparseArrays.sparse!(asm.matrix_pattern, storage)
end

function function_space(assembler::AbstractAssembler)
  return function_space(assembler.dof)
end

"""
$(TYPEDSIGNATURES)
"""
function hessian(asm::AbstractAssembler)
  backend = KA.get_backend(asm)
  H = _csc_matrix(asm, asm.hessian_storage)

  if _is_condensed(asm.dof)
    _adjust_matrix_entries_for_constraints!(H, asm.constraint_storage, backend)
  end

  return H
end

# new approach requiring access to the v that makes Hv
"""
$(TYPEDSIGNATURES)
"""
function hvp(asm::AbstractAssembler, v)
  if _is_condensed(asm.dof)
    _adjust_matrix_action_entries_for_constraints!(
      asm.stiffness_action_storage, asm.constraint_storage, v
    )
    return asm.stiffness_action_storage.data
  else
    extract_field_unknowns!(
      asm.stiffness_action_unknowns,
      asm.dof,
      asm.stiffness_action_storage
    )
    return asm.stiffness_action_unknowns
  end
end

"""
$(TYPEDSIGNATURES)
"""
function mass(asm::AbstractAssembler)
  backend = KA.get_backend(asm)
  M = _csc_matrix(asm, asm.mass_storage)

  if _is_condensed(asm.dof)
    _adjust_matrix_entries_for_constraints!(M, asm.constraint_storage, backend)
  end

  return M
end

"""
$(TYPEDSIGNATURES)
assumes assemble_vector! has already been called
"""
function residual(asm::AbstractAssembler; use_sparse_vector = false)
  if use_sparse_vector
    return sparsevec(asm.vector_pattern, asm.residual_unknowns)
  else
    if _is_condensed(asm.dof)
      _adjust_vector_entries_for_constraints!(
        asm.residual_storage, asm.constraint_storage
      )
      return asm.residual_storage.data
    else
      extract_field_unknowns!(
        asm.residual_unknowns, 
        asm.dof, 
        asm.residual_storage
      )
      return asm.residual_unknowns
    end
  end
end

"""
$(TYPEDSIGNATURES)
"""
function stiffness(asm::AbstractAssembler)
  backend = KA.get_backend(asm)
  K = _csc_matrix(asm, asm.stiffness_storage)

  if _is_condensed(asm.dof)
    _adjust_matrix_entries_for_constraints!(K, asm.constraint_storage, backend)
  end

  return K
end

function _assemble_block!(
  field,
  conns::Conn, coffset::Int,
  block_start_index::Int,
  func::Function,
  physics::AbstractPhysics, ref_fe::ReferenceFE,
  X::AbstractField, t::T, dt::T,
  U::Solution, U_old::Solution, 
  state_old::S, state_new::S, props::AbstractArray,
  return_type::R
) where {
  T        <: Number,
  Conn     <: AbstractArray,
  Solution <: AbstractField,
  S,       #<: L2QuadratureField
  R        <: AssembledReturnType
}
  fec_foraxes(state_old, 3) do e
    conn = connectivity(ref_fe, conns, e, coffset)
    x_el, u_el, u_el_old = element_level_fields(ref_fe, conn, X, U, U_old)
    props_el = _element_level_properties(props, e)
    val_el = _element_scratch(return_type, ref_fe, U)
    for q in 1:num_cell_quadrature_points(ref_fe)
      interps = _cell_interpolants(ref_fe, q)
      state_old_q = _quadrature_level_state(state_old, q, e)
      state_new_q = _quadrature_level_state(state_new, q, e)
      val_q = func(physics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el)
      val_el = _accumulate_q_value(return_type, field, val_q, val_el, q, e)
    end
  
    _assemble_element!(field, val_el, conn, e, block_start_index)
  end
end

function create_assembler_cache(
  asm::AbstractAssembler, 
  ::AssembledMatrix
)
  return zeros(length(asm.matrix_pattern.Is))
end

function create_assembler_cache(
  asm::AbstractAssembler,
  ::AssembledScalar
)
  fspace = function_space(asm.dof)
  field = L2Field(undef, Float64, 1, block_quadrature_sizes(fspace))
  fill!(field, 0.0)
  return field
end

function create_assembler_cache(
  asm::AbstractAssembler, 
  ::AssembledVector
)
  return create_field(asm)
end

# some utilities
include("Constraints.jl")
include("SparsityPatterns.jl")

# types
include("MatrixFreeAssembler.jl")
include("SparseMatrixAssembler.jl")
# include("SparseMatrixAssemblerNew.jl")

# methods
include("Matrix.jl")
include("MatrixAction.jl")
include("QuadratureQuantity.jl")
include("Source.jl")
include("Vector.jl")
include("WeaklyEnforcedBCs.jl")
