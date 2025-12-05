"""
$(TYPEDEF)
$(TYPEDFIELDS)
"""
abstract type AbstractAssembler{Dof <: DofManager} end
"""
$(TYPEDSIGNATURES)
"""
KA.get_backend(asm::AbstractAssembler) = KA.get_backend(asm.dof)

"""
$(TYPEDSIGNATURES)
Assembly method for an H1Field, e.g. internal force
Called on a single element e for a given block b where local_val
has already been constructed from quadrature contributions.
"""
function _assemble_element!(global_val::H1Field, local_val, conn)
  n_dofs = size(global_val, 1)
  for i in axes(conn, 1)
    for d in 1:n_dofs
      global_id = n_dofs * (conn[i] - 1) + d
      local_id = n_dofs * (i - 1) + d
      global_val[global_id] += local_val[local_id]
    end
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
