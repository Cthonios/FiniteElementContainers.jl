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

struct AssembledMatrix <: AssembledReturnType
end

struct AssembledScalar <: AssembledReturnType
end

struct AssembledStruct{T} <: AssembledReturnType
end

struct AssembledVector <: AssembledReturnType
end

# fall back method does nothing
@inline function _accumulate_q_value(::AssembledReturnType, storage, val_q, val_e, q, e)
  val_e = val_e + val_q
  return val_e
end

@inline function _accumulate_q_value(::AssembledScalar, storage, val_q, val_e, q, e)
  storage[q, e] = val_q
  return nothing
end

@inline function _accumulate_q_value(::AssembledStruct, storage, val_q, val_e, q, e)
  storage[q, e] = val_q
  return nothing
end

# assembly helper methods below
function _assemble_element!(
  storage, R_el::SVector, 
  conns, # all connectivities for this block
  el_id::Int, ::Int, ::Int
)
  n_dofs = size(storage, 1)
  for d in axes(storage, 1)
    for n in axes(conns, 1)
      global_id = n_dofs * (conns[n, el_id] - 1) + d
      local_id = n_dofs * (n - 1) + d
      Atomix.@atomic storage.data[global_id] += R_el[local_id]
    end
  end
  return nothing
end

# has a pretty dumb name for what it does
function _assemble_element!(
  storage, val_q::T, 
  conns, # all connectivities for this block
  ::Int, ::Int, ::Int
) where T <: Number
  return nothing
end

# TODO we'll need a regular matrix implementation
# as well (Can we live with 1?)
"""
$(TYPEDSIGNATURES)
Specialization of of ```_assemble_element!``` for ```SparseMatrixAssembler```.
"""
function _assemble_element!(
  storage, K_el::SMatrix, 
  conns, # all connectivities for this block
  el_id::Int,
  block_start_index::Int, block_el_level_size::Int
)
  # figure out ids needed to update
  start_id = block_start_index + 
             (el_id - 1) * block_el_level_size
  end_id = start_id + block_el_level_size - 1
  ids = start_id:end_id

  # get appropriate storage and update values
  # @views storage[ids] += K_el[:]
  for (i, id) in enumerate(ids)
    storage[id] += K_el.data[i]
  end
  return nothing
end

# has a pretty dumb name for what it does
# fall back fro struct
function _assemble_element!(
  storage, val_q, 
  conns, # all connectivities for this block
  ::Int, ::Int, ::Int
)
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

"""
$(TYPEDSIGNATURES)
"""
@inline function _element_level_fields(U::H1Field, ref_fe, conns, e)
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
  u_el = @views SMatrix{NNPE, ND, eltype(U), NxNDof}(U[:, conns[:, e]])
  return u_el
end

"""
$(TYPEDSIGNATURES)
"""
@inline function _element_level_fields_flat(U::H1Field, ref_fe, conns, e)
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
  u_el = @views SVector{NxNDof, eltype(U)}(U[:, conns[:, e]])
  return u_el
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
@inline function _element_level_properties(props::L2ElementField, e::Int)
  props_e = @views SVector{size(props, 1), eltype(props)}(props[:, e])
  return props_e
end

"""
$(TYPEDSIGNATURES)
"""
@inline function _element_scratch_matrix(ref_fe, U)
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
  K_el = zeros(SMatrix{NxNDof, NxNDof, eltype(U), NxNDof * NxNDof})
  return K_el
end

"""
$(TYPEDSIGNATURES)
"""
@inline function _element_scratch_vector(ref_fe, U)
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
  R_el = zeros(SVector{NxNDof, eltype(U)})
  return R_el
end

"""
$(TYPEDSIGNATURES)
"""
@inline function _element_scratch(::AssembledMatrix, ref_fe, U)
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
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
@inline function _element_scratch(::AssembledVector, ref_fe, U)
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
  R_el = zeros(SVector{NxNDof, eltype(U)})
  return R_el
end

"""
$(TYPEDSIGNATURES)
"""
function _quadrature_level_state(state::AbstractArray{<:Number, 3}, q::Int, e::Int)
  # NS = size(state, 1)
  # if NS > 0
  #   state_q = @views SVector{size(state, 1), eltype(state)}(state[:, q, e])
  # else
  #   state_q = SVector{0, eltype(state)}()
  # end
  # return state_q
  # NS = size(state, 1)
  state_q = view(state, :, q, e)
  return state_q
end

"""
$(TYPEDSIGNATURES)
"""
function hessian(asm::AbstractAssembler)
  return _hessian(asm, KA.get_backend(asm))
end

# new approach requiring access to the v that makes Hv
"""
$(TYPEDSIGNATURES)
"""
function hvp(asm::AbstractAssembler, v)
  if _is_condensed(asm.dof)
    _adjust_matrix_action_entries_for_constraints!(
      asm.stiffness_action_storage, asm.constraint_storage, v,
      KA.get_backend(asm)
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
function mass(assembler::AbstractAssembler)
  return _mass(assembler, KA.get_backend(assembler))
end

"""
$(TYPEDSIGNATURES)
assumes assemble_vector! has already been called
"""
function residual(asm::AbstractAssembler)
  if _is_condensed(asm.dof)
    _adjust_vector_entries_for_constraints!(
      asm.residual_storage, asm.constraint_storage,
      KA.get_backend(asm)
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

"""
$(TYPEDSIGNATURES)
"""
function stiffness(assembler::AbstractAssembler)
  return _stiffness(assembler, KA.get_backend(assembler))
end

# General assembler methods

# CPU implementation
"""
$(TYPEDSIGNATURES)
Assembly method for a block labelled as block_id. This is a CPU implementation
with no threading.

TODO add state variables and physics properties
"""
function _assemble_block!(
  ::KA.CPU,
  # field::AbstractArray{<:Number, 1}, 
  field,
  conns::Conn, block_start_index::Int, block_el_level_size::Int,
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

  for e in axes(conns, 2)
    x_el = _element_level_fields_flat(X, ref_fe, conns, e)
    u_el = _element_level_fields_flat(U, ref_fe, conns, e)
    u_el_old = _element_level_fields_flat(U_old, ref_fe, conns, e)
    props_el = _element_level_properties(props, e)
    val_el = _element_scratch(return_type, ref_fe, U)

    for q in 1:num_quadrature_points(ref_fe)
      interps = _cell_interpolants(ref_fe, q)
      state_old_q = _quadrature_level_state(state_old, q, e)
      state_new_q = _quadrature_level_state(state_new, q, e)
      val_q = func(physics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el)
      val_el = _accumulate_q_value(return_type, field, val_q, val_el, q, e)
    end
    _assemble_element!(field, val_el, conns, e, block_start_index, block_el_level_size)
  end
  return nothing
end

# GPU implementation
# COV_EXCL_START
KA.@kernel function _assemble_block_kernel!(
  # field::AbstractArray{<:Number, 1}, 
  field,
  conns::Conn, block_start_index::Int, block_el_level_size::Int,
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
  E = KA.@index(Global)

  x_el = _element_level_fields_flat(X, ref_fe, conns, E)
  u_el = _element_level_fields_flat(U, ref_fe, conns, E)
  u_el_old = _element_level_fields_flat(U_old, ref_fe, conns, E)
  props_el = _element_level_properties(props, E)
  val_el = _element_scratch(return_type, ref_fe, U)
  for q in 1:num_quadrature_points(ref_fe)
    interps = _cell_interpolants(ref_fe, q)
    state_old_q = _quadrature_level_state(state_old, q, E)
    state_new_q = _quadrature_level_state(state_new, q, E)
    val_q = func(physics, interps, x_el, t, dt, u_el, u_el_old, state_old_q, state_new_q, props_el)
    val_el = _accumulate_q_value(return_type, field, val_q, val_el, q, E)
  end

  _assemble_element!(field, val_el, conns, E, block_start_index, block_el_level_size)
end
# COV_EXCL_STOP

# method for kernel generation
function _assemble_block!(
  backend::KA.Backend, 
  field, 
  conns, block_start_index, block_el_level_size,
  func,
  physics, ref_fe,
  X, t, dt, U, U_old, state_old, state_new, props,
  return_type
)
  kernel! = _assemble_block_kernel!(backend)
  kernel!(
    field,
    conns, block_start_index, block_el_level_size, 
    func,
    physics, ref_fe,
    X, t, dt,
    U, U_old, 
    state_old, state_new, props,
    return_type,
    ndrange=size(conns, 2)
  )
  return nothing
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
  vals = L2ElementField[]
  fspace = function_space(asm.dof)
  for (conn, ref_fe) in zip(values(fspace.elem_conns), values(fspace.ref_fes))
    NQ = ReferenceFiniteElements.num_quadrature_points(ref_fe)
    NE = size(conn, 2)
    push!(vals, L2ElementField(zeros(Float64, NQ, NE)))
  end
  return NamedTuple{keys(fspace.elem_conns)}(tuple(vals...))
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
include("NeumannBC.jl")
include("QuadratureQuantity.jl")
include("Vector.jl")
