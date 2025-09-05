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
function _assemble_element!(global_val::H1Field, local_val, conn, e, b)
  n_dofs = size(global_val, 1)
  for i in axes(conn, 1)
    for d in 1:n_dofs
      # n = 2 * i + d
      global_id = n_dofs * (conn[i] - 1) + d
      local_id = n_dofs * (i - 1) + d
      global_val[global_id] += local_val[local_id]
    end
  end
  return nothing
end

function _check_backends(assembler, U, X, state_old, state_new, conns)
  backend = KA.get_backend(assembler)
  # TODO add get_backend method of ref_fe
  @assert backend == KA.get_backend(U)
  @assert backend == KA.get_backend(X)
  @assert backend == KA.get_backend(conns)
  @assert backend == KA.get_backend(state_old)
  @assert backend == KA.get_backend(state_new)
  # props will be complicated...
  # TODO
  return backend
end

function _check_backends(assembler, U, V, X, state_old, state_new, conns)
  backend = KA.get_backend(assembler)
  # TODO add get_backend method of ref_fe
  @assert backend == KA.get_backend(U)
  @assert backend == KA.get_backend(V)
  @assert backend == KA.get_backend(X)
  @assert backend == KA.get_backend(conns)
  @assert backend == KA.get_backend(state_old)
  @assert backend == KA.get_backend(state_new)
  # props will be complicated...
  # TODO
  return backend
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
function _quadrature_level_state(state::L2QuadratureField, q::Int, e::Int)
  NS = size(state, 1)
  if NS > 0
    state_q = @views SVector{size(state, 1), eltype(state)}(state[:, q, e])
  else
    state_q = SVector{0, eltype(state)}()
  end
  return state_q
end

"""
$(TYPEDSIGNATURES)
"""
function hvp(assembler::AbstractAssembler)
  extract_field_unknowns!(
    assembler.stiffness_action_unknowns,
    assembler.dof,
    assembler.stiffness_action_storage
  )
  return assembler.stiffness_action_unknowns
end

"""
$(TYPEDSIGNATURES)
"""
function mass(assembler::AbstractAssembler)
  return _mass(assembler, KA.get_backend(assembler))
end

"""
$(TYPEDSIGNATURES)
"""
function residual(asm::AbstractAssembler)
  extract_field_unknowns!(
    asm.residual_unknowns, 
    asm.dof, 
    asm.residual_storage
  )
  return asm.residual_unknowns
end

"""
$(TYPEDSIGNATURES)
"""
function stiffness(assembler::AbstractAssembler)
  return _stiffness(assembler, KA.get_backend(assembler))
end

# some utilities
include("SparsityPattern.jl")

# types
include("MatrixFreeAssembler.jl")
include("SparseMatrixAssembler.jl")

# methods
include("Matrix.jl")
include("MatrixAction.jl")
include("NeumannBC.jl")
include("QuadratureQuantity.jl")
include("Vector.jl")
