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
Top level assembly method for ```H1Field``` that loops over blocks and dispatches
to appropriate kernels based on sym.

TODO need to make sure at setup time that physics and elem_conns have the same
values order. Otherwise, shenanigans.

TODO figure out how to do generated functions

creates one type instability from the Val
"""
function assemble!(assembler, Uu, p, sym::Symbol, type::Type{H1Field})
  assemble!(assembler, Uu, p, Val{sym}(), type)
end

"""
$(TYPEDSIGNATURES)
Top level assembly method for ```H1Field``` that loops over blocks and dispatches
to appropriate kernels based on sym.

TODO need to make sure at setup time that physics and elem_conns have the same
values order. Otherwise, shenanigans.

TODO figure out how to do generated functions

creates one type instability from the Val
"""
function assemble!(assembler, Uu, p, Vv, sym::Symbol, type::Type{H1Field})
  assemble!(assembler, Uu, p, Vv, Val{sym}(), type)
end

"""
$(TYPEDSIGNATURES)
"""
@inline function _cell_interpolants(ref_fe::R, q::Int) where R <: ReferenceFE
  return @inbounds ref_fe.cell_interps.vals[q]
end

"""
$(TYPEDSIGNATURES)
"""
create_field(asm::AbstractAssembler, type) = create_field(asm.dof, type)

"""
$(TYPEDSIGNATURES)
"""
create_unknowns(asm::AbstractAssembler, type::Type{<:AbstractField}) = create_unknowns(asm.dof, type)

"""
$(TYPEDSIGNATURES)
Should we keep this?
"""
function _element_level_fields(U::H1Field, ref_fe, conns, e)
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
  u_el = @views SMatrix{ND, NNPE, Float64, NxNDof}(U[:, conns[:, e]])
  return u_el
end

function _element_level_fields_flat(U::H1Field, ref_fe, conns, e)
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
  u_el = @views SVector{NxNDof, eltype(U)}(U[:, conns[:, e]])
  return u_el
end

"""
$(TYPEDSIGNATURES)
"""
function _element_level_properties(props::SVector{NP, T}, ::Int) where {NP, T}
  return props
end

"""
$(TYPEDSIGNATURES)
"""
function _element_level_properties(props::L2ElementField, e::Int)
  props_e = @views SVector{size(props, 1), eltype(props)}(props[:, e])
  return props_e
end

"""
$(TYPEDSIGNATURES)
Returns either nothing or the function space
associated with H1Fields in the problem.
"""
@inline function function_space(asm::AbstractAssembler, ::Type{<:H1Field})
  if asm.dof.H1_vars === nothing
    @assert "You're trying to access a function space that does not exist"
  end
  return @inbounds asm.dof.H1_vars[1].fspace
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
  return _hvp(assembler, KA.get_backend(assembler))
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
  return _residual(asm, KA.get_backend(asm))
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
include("SparseMatrixAssembler.jl")

# methods
include("Matrix.jl")
include("MatrixAction.jl")
include("MatrixAndVector.jl")
include("NeumannBC.jl")
include("Scalar.jl")
include("Vector.jl")
