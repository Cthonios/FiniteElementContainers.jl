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
Assembly method for a scalar field stored as a size 1 vector
Called on a single element e for a given block b where local_val
has already been constructed from quadrature contributions.
"""
function _assemble_element!(global_val::T, local_val, conn, e, b) where T <: AbstractArray{<:Number, 1}
  global_val[1] += local_val
  return nothing
end

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

_assemble_block_method_from_sym(::Val{:mass}) = _assemble_block_mass!
_assemble_block_method_from_sym(::Val{:residual}) = _assemble_block_residual!
_assemble_block_method_from_sym(::Val{:residual_and_stiffness}) = _assemble_block_residual_and_stiffness!
_assemble_block_method_from_sym(::Val{:stiffness}) = _assemble_block_stiffness!

function _check_backends(assembler, U, X, conns)
  backend = KA.get_backend(assembler)
  # TODO add get_backend method of ref_fe
  @assert backend == KA.get_backend(U)
  @assert backend == KA.get_backend(X)
  @assert backend == KA.get_backend(conns)
  return backend
end

"""
$(TYPEDSIGNATURES)
Top level assembly method for ```H1Field``` that loops over blocks and dispatches
to appropriate kernels based on sym.

TODO need to make sure at setup time that physics and elem_conns have the same
values order. Otherwise, shenanigans.
"""
function assemble!(assembler, physics, U::H1Field, sym)
  val_sym = Val{sym}()
  _assemble_block_method! = _assemble_block_method_from_sym(val_sym)
  _zero_storage(assembler, val_sym)
  fspace = assembler.dof.H1_vars[1].fspace
  X = fspace.coords
  for (b, (conns, block_physics)) in enumerate(zip(values(fspace.elem_conns), values(physics)))
    ref_fe = values(fspace.ref_fes)[b]
    backend = _check_backends(assembler, U, X, conns)
    _assemble_block_method!(assembler, block_physics, ref_fe, U, X, conns, b, backend)
  end
end

"""
$(TYPEDSIGNATURES)
"""
create_bcs(asm::AbstractAssembler, type) = create_bcs(asm.dof, type)
"""
$(TYPEDSIGNATURES)
"""
create_field(asm::AbstractAssembler, type) = create_field(asm.dof, type)
"""
$(TYPEDSIGNATURES)
"""
create_unknowns(asm::AbstractAssembler) = create_unknowns(asm.dof)

"""
$(TYPEDSIGNATURES)
"""
create_unknowns(asm::AbstractAssembler, type::Type{<:AbstractField}) = create_unknowns(asm.dof, type)

"""
$(TYPEDSIGNATURES)
"""
function _element_level_fields(U::H1Field, ref_fe, conns, e)
  ND = size(U, 1)
  NNPE = ReferenceFiniteElements.num_vertices(ref_fe)
  NxNDof = NNPE * ND
  u_el = @views SMatrix{ND, NNPE, Float64, NxNDof}(U[:, conns[:, e]])
  return u_el
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
function update_field!(U, asm::AbstractAssembler, Uu, Ubc)
  update_field!(U, asm.dof, Uu, Ubc)
  return nothing
end

"""
$(TYPEDSIGNATURES)
"""
function _zero_storage(asm::AbstractAssembler, ::Val{:residual})
  fill!(asm.residual_storage.vals, zero(eltype(asm.residual_storage.vals)))
end

# different backend implementations of abstract methods
include("CPUGeneral.jl")
include("GPUGeneral.jl")

# some utilities
include("SparsityPattern.jl")

# implementations
include("SparseMatrixAssembler.jl")
